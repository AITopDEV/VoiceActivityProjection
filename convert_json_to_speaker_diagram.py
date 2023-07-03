import json
import datetime

data = json.load(open('banana.json'))
p_now = data['p_now'][0]
text_file = "next_speaker_probability.txt"

current = datetime.time(0, 0, 0, 0)
time_delta = datetime.timedelta(milliseconds=20)
file = open(text_file, "wt")
file.write("\tSTART\t\t\tEND\t\t\t    SPEAKER_1\t\t\tSPEAKER_2\n")
for i in range(len(p_now)):
    end = (datetime.datetime.combine(datetime.date.today(), current) + time_delta).time()
    line = f"{current.strftime('%H:%M:%S.%f')[:-3]} : {end.strftime('%H:%M:%S.%f')[:-3]} --- {p_now[i][0]}\t{p_now[i][1]}\n"
    file.write(line)
    current = end
file.close()