import torch
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
import numpy as np


if torch.cuda.is_available():
    print("Cuda is available")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

BATCH_SIZE = 1000

print('===> Loading datasets')

train_set = get_training_set()
training_data_loader = DataLoader(dataset=train_set, num_workers=7, batch_size=BATCH_SIZE, shuffle=False)

#train_set = get_test_set()
#training_data_loader = DataLoader(dataset=train_set, num_workers=7, batch_size=BATCH_SIZE, shuffle=False)

#means = np.zeros((7,3))
#squaredMeans = np.zeros((7,3))

means = np.zeros((7,3))
squaredMeans = np.zeros((7,3))

for iteration, batch in enumerate(training_data_loader, 1):
    input, target = batch[0].to(device), batch[1].to(device)
    print(iteration)
    print(input.shape, target.shape)
    inputSquared = input**2
    means[iteration-1] = torch.mean(input, dim=(0,2,3)).numpy()
    squaredMeans[iteration-1] = torch.mean(inputSquared, dim=(0,2,3)).numpy()


#print(means)
#print(squaredMeans)

meanedMeans = means.mean(axis=0)
Variances = squaredMeans.mean(axis=0)-meanedMeans**2

lastArr = np.concatenate(([meanedMeans], [Variances]), axis=0)
print(lastArr)
np.save("calcMeanStddiv/separatedStatistics2.npy", lastArr)
exit()

BATCH_SIZE = 500

print('===> Loading datasets')

#train_set = get_training_set()
#training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)

train_set = get_test_set()
training_data_loader = DataLoader(dataset=train_set, num_workers=7, batch_size=BATCH_SIZE, shuffle=False)

#means = np.zeros((7,3))
#squaredMeans = np.zeros((7,3))

mean = np.zeros(3)
variance = np.zeros(3)

for iteration, batch in enumerate(training_data_loader, 1):
    input, target = batch[0].to(device), batch[1].to(device)
    print(iteration)
    print(input.shape, target.shape)
    mean = torch.mean(input, dim=(0,2,3)).numpy()
    variance = torch.var(input, dim=(0,2,3)).numpy()




lastArr = np.concatenate(([mean], [variance]), axis=0)
print(lastArr)
np.save("calcMeanStddiv/fullStatistics2.npy", lastArr)

