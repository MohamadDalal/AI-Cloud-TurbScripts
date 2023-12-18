import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt("avg_loss.csv")
test = np.loadtxt("validation_avg_loss.csv")

plt.plot(range(1,len(train)+1), train, label="train")
plt.plot(range(1,len(test)+1), test, label="validation")
plt.legend()
plt.title("Average MSE loss in training")
plt.xlabel("Epochs")
plt.ylabel("MSE")
#plt.hlines([0.002,0.0055], 0, 30, color="k", linewidth=1)
#plt.vlines([0,30], 0.002, 0.0055, color="k", linewidth=1)
plt.savefig("mse.png", dpi=400)
plt.ylim((0.0019, 0.005))
plt.savefig("mse-zoomed.png", dpi=400)
plt.show()
