import numpy as np
from sklearn.model_selection import train_test_split
import os


splits = np.array([0.7, 0.2, 0.1])
all_file_list = os.listdir("/home/student.aau.dk/mdalal20/channelData")
#all_file_list = os.listdir("/home/mdalal/Documents/AiCloudFiles/dataSent/")
file_list = [x.split(".")[0] for x in all_file_list if x.split(".")[-1] == "h5"]
print(file_list)

RNG = np.random.default_rng(1234)
permuted_list = RNG.permutation(file_list)

splits = splits.cumsum()*len(permuted_list)
train, validation, test = np.split(permuted_list, splits[:-1].astype(int))

print(train, validation, test)
np.savetxt("training_chunks.csv", train, fmt="%s")
np.savetxt("validation_chunks.csv", validation, fmt="%s")
np.savetxt("test_chunks.csv", test, fmt="%s")