import numpy as np
#
a = ['0.05', '0.1', '0.25', '0.5', '0.75', '0.8']
# np.savez('patrial_node.npz', a)
#
# data = np.load('patrial_node.npz', allow_pickle=True)
# print(data.files)

import random
random.seed(0)
selected = sorted(random.sample(range(0,207), 52))


print(selected)
print(selected.sort())