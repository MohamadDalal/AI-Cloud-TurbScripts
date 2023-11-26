import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mse = pd.read_csv("test_mse.csv")
psnr = pd.read_csv("test_psnr.csv")

#plt.plot(range(1,len(train)+1), train, label="train")
#plt.plot(range(1,len(test)+1), test, label="validation")
#plt.legend()
#plt.title("Average MSE loss in training")
#plt.xlabel("Epochs")
#plt.ylabel("MSE")
#plt.savefig("mse.png")
#plt.show()