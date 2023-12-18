import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt("avg_mse.csv")
test = np.loadtxt("validation_avg_mse.csv")

plt.plot(range(2,len(train)+1), train[1:], label="train")
plt.plot(range(2,len(test)+1), test[1:], label="validation")
plt.legend()
plt.title("Average MSE loss in training")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.savefig("mse2.png", dpi=400)
plt.show()

