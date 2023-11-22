from time import perf_counter
from data import get_test_set
from torch.utils.data import DataLoader
import torch

if torch.cuda.is_available():
    print("Cuda is available")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

for num_workers in range(4, 33, 4):  
    dataset = get_test_set()
    #print(num_workers)
    test_data_loader = DataLoader(dataset=dataset, num_workers=4, batch_size=32 , shuffle=False)
    start = perf_counter()
    for epoch in range(1, 3):
        for batch in test_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)
        #print(epoch)
    end = perf_counter()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))