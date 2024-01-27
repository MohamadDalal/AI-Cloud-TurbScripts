import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Liberation Serif"

train = np.loadtxt("avg_loss.csv")
test = np.loadtxt("validation_avg_loss.csv")

plt.plot(range(1,len(train)+1), train, label="train")
plt.plot(range(1,len(test)+1), test, label="validation")
plt.legend()
plt.title("Average MSE loss in training")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.hlines([0.001,0.004], 0, 125, color="k", linewidth=0.5)
plt.vlines([0,125], 0.001, 0.004, color="k", linewidth=0.5)
plt.savefig("mse3.png", dpi=400, bbox_inches='tight')
plt.cla()

plt.plot(range(1,len(train)+1), train, label="train")
plt.plot(range(1,len(test)+1), test, label="validation")
plt.ylim((0.002, 0.004))
plt.savefig("mse3-zoomed.png", dpi=400, bbox_inches='tight')
plt.show()
