import numpy as np
import matplotlib.pyplot as plt



for i in range(3):
    Mean = np.load(f"MAE_Sum.npy")[...,i]/1000
    Var = np.load(f"VarChannel{i}.npy")
    Fig, ax = plt.subplots()
    pcm = ax.imshow(Var/Mean)
    Fig.colorbar(pcm, ax=ax)
    Fig.savefig(f"RatioChannel{i}.png", dpi=400, bbox_inches='tight')