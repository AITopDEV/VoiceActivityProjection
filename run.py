from argparse import ArgumentParser
from os.path import basename
import matplotlib.pyplot as plt
import torch
import torchaudio
import time
import datetime
from vap.model import VapGPT, VapConfig, load_older_state_dict
from vap.audio import load_waveform
from vap.utils import (
    batch_to_device,
    everything_deterministic,
    tensor_dict_to_json,
    write_json,
)
from vap.plot_utils import plot_stereo


everything_deterministic()
torch.manual_seed(0)


def step_extraction(
    waveform,
    model,
    device="cpu",
    context_time=20,
    step_time=5,
    vad_thresh=0.5,
    ipu_time=0.1,
    pbar=True,
    verbose=False,
):
    """
    Takes a waveform, the model, and extracts probability output in chunks with
    a specific context and step time. Concatenates the output accordingly and returns full waveform output.
    """

    n_samples = waveform.shape[-1]
    duration = round(n_samples / model.sample_rate, 2)

    chunk_time = 5

    # Samples
    # context_samples = int(context_time * model.sample_rate)
    step_samples = int(step_time * model.sample_rate)
    chunk_samples = int(chunk_time * model.sample_rate)
    chunk_frames = int(chunk_time * model.frame_hz)
    step_frames = int(step_time * model.frame_hz)

    # Fold the waveform to get total chunks
    folds = waveform.unfold(
        dimension=-1, size=chunk_samples, step=step_samples
    ).permute(2, 0, 1, 3)
    print("folds: ", tuple(folds.shape))

    expected_frames = round(duration * model.frame_hz)
    n_folds = int((n_samples - chunk_samples) / step_samples + 1.0)
    total = (n_folds - 1) * step_samples + chunk_samples

    # First chunk
    # Use all extracted data. Does not overlap with anything prior.
    out = model.probs(folds[0].to(device))

    if pbar:
        from tqdm import tqdm

        pbar = tqdm(folds[1:], desc=f"Context: {context_time}s, step: {step_time}")
    else:
        pbar = folds[1:]
    for w in pbar:
        o = model.probs(w.to(device))
        out["vad"] = torch.cat([out["vad"], o["vad"][:, -step_frames:]], dim=1)
        out["p_now"] = torch.cat([out["p_now"], o["p_now"][:, -step_frames:]], dim=1)
        out["p_future"] = torch.cat(
            [out["p_future"], o["p_future"][:, -step_frames:]], dim=1
        )
        out["probs"] = torch.cat([out["probs"], o["probs"][:, -step_frames:]], dim=1)
        out["H"] = torch.cat([out["H"], o["H"][:, -step_frames:]], dim=1)

    processed_frames = out["p_now"].shape[1]

    ###################################################################
    # Handle LAST SEGMENT (not included in `unfold`)
    ###################################################################
    if expected_frames != processed_frames:
        omitted_frames = expected_frames - processed_frames

        omitted_samples = model.sample_rate * omitted_frames / model.frame_hz

        if verbose:
            print(f"Expected frames {expected_frames} != {processed_frames}")
            print(f"omitted frames: {omitted_frames}")
            print(f"omitted samples: {omitted_samples}")
            print(f"chunk_samples: {chunk_samples}")

        w = waveform[..., -chunk_samples:]
        o = model.probs(w.to(device))
        out["vad"] = torch.cat([out["vad"], o["vad"][:, -omitted_frames:]], dim=1)
        out["p_now"] = torch.cat([out["p_now"], o["p_now"][:, -omitted_frames:]], dim=1)
        out["p_future"] = torch.cat(
            [out["p_future"], o["p_future"][:, -omitted_frames:]], dim=1
        )
        out["probs"] = torch.cat([out["probs"], o["probs"][:, -omitted_frames:]], dim=1)
        out["H"] = torch.cat([out["H"], o["H"][:, -omitted_frames:]], dim=1)
    out = batch_to_device(out, "cpu")  # to cpu for plot/save
    return out


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-a",
        "--audio",
        type=str,
        help="Path to waveform",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default=None,
        help="Path to waveform",
    )
    parser.add_argument(
        "-sd",
        "--state_dict",
        type=str,
        default="example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt",
        help="Path to state_dict",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model",
    )
    parser, _ = VapConfig.add_argparse_args(parser)
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Process the audio in chunks (longer > 164s on 24Gb GPU audio)",
    )
    parser.add_argument(
        "--chunk_time",
        type=float,
        default=5,
        help="Duration of each chunk processed by model",
    )
    parser.add_argument(
        "--step_time",
        type=float,
        default=0.5,
        help="Increment to process in a step",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Visualize output (matplotlib)"
    )
    args = parser.parse_args()

    conf = VapConfig.args_to_conf(args)
    return args, conf


if __name__ == "__main__":
    args, conf = get_args()

    ###########################################################
    # Load the model
    ###########################################################
    print("Load Model...")
    if args.checkpoint is None:
        print("From state-dict: ", args.state_dict)
        model = VapGPT(conf)
        sd = torch.load(args.state_dict)
        model.load_state_dict(sd)
    else:
        from vap.train import VAPModel

        print("From Lightning checkpoint: ", args.checkpoint)
        raise NotImplementedError("Not implemeted from checkpoint...")
        # model = VAPModel.load_from_checkpoint(args.checkpoint)
    device = "cpu"
    if torch.cuda.is_available():
        model = model.to("cuda")
        device = "cuda"
    model = model.eval()

    ###########################################################
    # Load the Audio
    ###########################################################
    orig_waveform, _ = load_waveform(args.audio, sample_rate=model.sample_rate)
    # waveform = waveform[:, :50000]
    duration = round(orig_waveform.shape[-1] / model.sample_rate)
    start_time = 0
    STEP = 0.5


    chunk_size = 5
    vad_array = []
    p_ns_array = []
    current = datetime.time(0, 0, 0, 0)
    time_delta = datetime.timedelta(milliseconds=20)
    while start_time < duration:
        s = time.time()
        if start_time + STEP > duration:
            end_time = duration
        else:
            end_time = start_time + STEP
        if end_time < chunk_size:
            waveform, _ = load_waveform(args.audio, sample_rate=model.sample_rate, start_time=0, end_time=end_time)
        else:
            waveform, _ = load_waveform(args.audio, sample_rate=model.sample_rate, start_time=end_time - chunk_size, end_time=end_time)

        if waveform.shape[0] == 1:
            waveform = torch.cat((waveform, torch.zeros_like(waveform)))
        waveform = waveform.unsqueeze(0)
        if torch.cuda.is_available():
            waveform = waveform.to("cuda")
        out = model.probs(waveform)
        out = batch_to_device(out, "cpu")
        vad_array.append(out["vad"][0][-26:-1].cpu())
        p_ns = out["p_now"][0, -26:-1, :].cpu()
        p_ns_array.append(p_ns[:, 0])
        if (p_ns > 0.5).all():
            speaker = "SPEAKER_1"
        elif (p_ns < 0.5).all():
            speaker = "SPEAKER_2"
        else:
            speaker = "UNKNOWN"
        for i in range(len(p_ns)):
            end = (datetime.datetime.combine(datetime.date.today(), current) + time_delta).time()
            line = f"{current.strftime('%H:%M:%S.%f')[:-3]} : {end.strftime('%H:%M:%S.%f')[:-3]} --- {p_ns[i][0]}\t{p_ns[i][1]}\n"
            print(line)
            current = end
        # print(int(start_time * 100) / 100, int(end_time * 100) / 100, speaker)
        start_time = end_time

    if args.plot:
        vad = torch.concat(vad_array, dim=0)
        p_ns = torch.concat(p_ns_array, dim=0)
        print(vad.shape)
        fig, ax = plot_stereo(
            orig_waveform.cpu(), p_ns, vad, plot=False, figsize=(100, 6)
        )
        # Save figure
        figpath = args.filename.replace(".json", ".png")
        fig.savefig(figpath)
        print(f"Saved figure as {figpath}.png")
        print("Close figure to continue")
        plt.show()
