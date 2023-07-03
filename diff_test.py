import json
import matplotlib.pyplot as plt
import numpy as np
data = json.load(open('banana.json'))
p_now = np.array(data['p_now'][0])
data = json.load(open('banana_1.json'))
p_now_1 = np.array(data['p_now'][0])
print(p_now.shape)
plt.plot(p_now[:, 0], marker='o')
plt.plot(p_now_1[51:, 0])
plt.show()
