import numpy as np
import matplotlib.pyplot as plt

Arr = np.load("MAE_Sum.npy")

for i in range(3):
    Fig, ax = plt.subplots()
    pcm = ax.imshow(Arr[...,i])
    Fig.colorbar(pcm, ax=ax)
    Fig.savefig(f"SumChannel{i}.png", dpi=400, bbox_inches='tight')