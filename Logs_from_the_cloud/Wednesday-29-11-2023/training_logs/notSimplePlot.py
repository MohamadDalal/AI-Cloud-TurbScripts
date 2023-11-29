import numpy as np
import matplotlib.pyplot as plt

"""
train = []
test = []

for i in range(1,31):
    train.append(np.loadtxt(f"Epoch_{i}/mse.csv"))
    test.append(np.loadtxt(f"Epoch_{i}/validation_mse.csv"))

print(np.shape(train))
print(np.shape(test))

train = np.reshape(train, np.shape(train)[0]*np.shape(train)[1])
"""
train = np.loadtxt("Epoch_1/mse.csv")

plt.plot(range(1,len(train)+1), train)
plt.title("MSE loss in first epoch")
plt.xlabel("Batch")
plt.ylabel("MSE")
#plt.hlines([0.002,0.0055], 0, 30, color="k", linewidth=1)
#plt.vlines([0,30], 0.002, 0.0055, color="k", linewidth=1)
plt.savefig("mse2.png", dpi=400)
#plt.ylim((0.0020, 0.0055))
#plt.savefig("mse-zoomed.png", dpi=400)
plt.show()

