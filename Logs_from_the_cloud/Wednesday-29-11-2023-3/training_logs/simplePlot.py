import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt("avg_mse.csv")
test = np.loadtxt("validation_avg_mse.csv")

plt.plot(range(1,len(train)+1), train, label="train")
plt.plot(range(1,len(test)+1), test, label="validation")
plt.legend()
plt.title("Average vector divergence loss in training")
plt.xlabel("Epochs")
plt.ylabel("Vector Divergence")
#plt.hlines([0.002,0.0055], 0, 30, color="k", linewidth=1)
#plt.vlines([0,30], 0.002, 0.0055, color="k", linewidth=1)
plt.savefig("mse.png", dpi=400)
plt.ylim((0.035, 0.05))
plt.savefig("mse-zoomed.png", dpi=400)
plt.show()
