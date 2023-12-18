import numpy as np
import matplotlib.pyplot as plt

Arr = np.load("MAE_Sum.npy")

for i in range(3):
    Mean = Arr[...,i]/1000
    Fig, ax = plt.subplots()
    pcm = ax.imshow(Mean)
    Fig.colorbar(pcm, ax=ax)
    Fig.savefig(f"MeanChannel{i}.png", dpi=400, bbox_inches='tight')