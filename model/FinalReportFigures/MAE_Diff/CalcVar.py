from __future__ import print_function
from os import getcwd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from os import listdir

def is_array_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])


def load_array(filepath):
    return np.load(filepath)

test_dir = join(getcwd(), "SingleDiff")
image_filenames = [x for x in sorted(listdir(test_dir)) if is_array_file(x)]

Channel = 0

BigBoi = np.zeros((1000,1536,512))

for i, x in enumerate(image_filenames):
    print(i)
    BigBoi[i] = np.load(f"{test_dir}/{x}")[...,Channel]

Variance = np.var(BigBoi, axis=0)
np.save(f"VarChannel{Channel}.npy", Variance)

Fig, ax = plt.subplots()

pcm = ax.imshow(Variance)
Fig.colorbar(pcm, ax=ax)
Fig.savefig(f"VarChannel{Channel}.png", dpi=400, bbox_inches='tight')