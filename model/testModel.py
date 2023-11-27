from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from os.path import join
from os import getcwd
from pathlib import Path
from model import Net
from data import get_test_set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter

def test(model, criterion):
    avg_psnr = 0
    psnr_list = []
    epoch_mse = 0
    mse_list = []
    with torch.no_grad():
        for index, batch in enumerate(testing_data_loader):
            input, target = batch[0].to(device), batch[1].to(device)
            #print(input.shape, target.shape)
            prediction = model(input)
            #print(prediction.shape)
            mse = criterion(prediction, target)
            batch_mse = mse.item()
            epoch_mse += batch_mse
            mse_list.append((test_set.image_filenames[index],batch_mse))
            psnr = 10 * log10(1 / batch_mse)
            avg_psnr += psnr
            psnr_list.append((test_set.image_filenames[index], psnr))
            #print(f"{test_set.image_filenames[index]}, {batch_mse}, {psnr}")
    print("===> Avg. MSE: {:.4f}".format(epoch_mse / len(testing_data_loader)))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    return epoch_mse, epoch_mse / len(testing_data_loader), mse_list, avg_psnr, avg_psnr / len(testing_data_loader), psnr_list

def log_seperate_epoch(test_mse, test_psnr, logging_path):
    save_path = Path(logging_path)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    #np.savetxt(f"{save_path}/test_mse.csv", test_mse)
    #np.savetxt(f"{save_path}/test_psnr.csv", test_psnr)
    pd.DataFrame(test_mse).to_csv(f"{save_path}/test_mse.csv", index=False)
    pd.DataFrame(test_psnr).to_csv(f"{save_path}/test_psnr.csv", index=False)

BATCH_SIZE = 1
#CHECKPOINT_PATH = "../Logs_from_the_cloud/Wednesday-22-11-2023/model_checkpoints/model_epoch_30.pth"
CHECKPOINT_PATH = "../Logs_from_the_cloud/Friday-24-11-2023-2/model_checkpoints/model_epoch_30.pth"
SAVE_PATH = "test_logs/"


if torch.cuda.is_available():
    print("Cuda is available")
    device = torch.device("cuda")
    checkpoint_dict = torch.load(CHECKPOINT_PATH)
else:
    device = torch.device("cpu")
    checkpoint_dict = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))

test_set = get_test_set()
testing_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)

model1 = Net(upscale_factor=32).to(device)
criterion_mae = nn.L1Loss()
criterion_mse = nn.MSELoss()
model1.load_state_dict(checkpoint_dict["model_state_dict"])

model1_start = perf_counter()
test_mse1, test_avg_mse1, test_iter_mse1, test_psnr1, test_avg_psnr1, test_iter_psnr1 = test(model1, criterion_mse)
model1_end = perf_counter()
print(f"Time taken for model 1 = {model1_end-model1_start}")

M1_Path = join(SAVE_PATH, "M1-B128-E30-L001-FixedResize-FixedGaussian-FixedDataloader")
log_seperate_epoch(test_iter_mse1, test_iter_psnr1, M1_Path)
with open(f"{M1_Path}/SingleResults.txt", "w") as f:
    f.write(f"TOT_MSE = {test_mse1}\nTOT_PSNR = {test_psnr1}\nAVG_MSE = {test_avg_mse1}\nAVG_PSNR = {test_avg_psnr1}\nTime_Taken = {model1_end-model1_start}s")

#exit()
model2 = lambda x : nn.functional.interpolate(x, (1536, 512), mode="bicubic")

model2_start = perf_counter()
test_mse2, test_avg_mse2, test_iter_mse2, test_psnr2, test_avg_psnr2, test_iter_psnr2 = test(model2, criterion_mse)
model2_end = perf_counter()
print(f"Time taken for model 2 = {model2_end-model2_start}")

M2_Path = join(SAVE_PATH, "Bicubic-FixedDataloader")
log_seperate_epoch(test_iter_mse2, test_iter_psnr2, M2_Path)
with open(f"{M2_Path}/SingleResults.txt", "w") as f:
    f.write(f"TOT_MSE = {test_mse2}\nTOT_PSNR = {test_psnr2}\nAVG_MSE = {test_avg_mse2}\nAVG_PSNR = {test_avg_psnr2}\nTime_Taken = {model2_end-model2_start}s")
