import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt("avg_mse.csv")
test = np.loadtxt("test_avg_mse_fixed.csv")

plt.plot(range(1,len(train)+1), train, label="train")
plt.plot(range(1,len(test)+1), test, label="validation")
plt.legend()
plt.title("Average MSE loss in training")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.savefig("mse.png")
plt.show()