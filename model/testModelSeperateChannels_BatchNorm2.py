from math import log10
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Resize, GaussianBlur 
from torch.utils.data import DataLoader
from os.path import join
from os import getcwd
from pathlib import Path
#from model import Net
from multiscale_model import Net
#from ESPCN_model import Net
#from old_model import Net
from dataset import DatasetFromFolder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter
from Custom_loss import div_loss

def input_transform():
    return Compose([
        ToTensor(),
        GaussianBlur(9, 1),
        #Resize((49,16), antialias=True),
        #Resize((48,16), antialias=True),
        Resize((192, 64), antialias=False),
    ])


def target_transform():
    return Compose([
        ToTensor(),
    ])

def get_test_set():
    root_dir = join(getcwd(), "49x18data", "all_data")
    test_dir = join(root_dir, "test")
    if torch.cuda.is_available():
        print("get_test_set: Cuda is available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return DatasetFromFolder(test_dir,
                             device,
                             input_transform=input_transform(),
                             target_transform=target_transform())


def test(model, channel):
    #TEST_Path = join(SAVE_PATH, "DSCMS-B128-E120-L0001-MSELoss-LastData_WithPredictions", f"Channel{testChannel}", "Test_Res")
    #Path(TEST_Path).mkdir(parents=True, exist_ok=True)
    avg_psnr = 0
    psnr_list = []
    epoch_mse = 0
    mse_list = []
    epoch_mae = 0
    mae_list = []
    epoch_div = 0
    div_list = []
    batchStatistic = torch.zeros((2,3))
    batchStatistic2 = torch.zeros((2,3))
    with torch.no_grad():
        # Split loss to each channel
        for index, batch in enumerate(testing_data_loader):
            input, target = batch[0].to(device), batch[1].to(device)
            batchStatistic[0] = torch.mean(input, dim=(0,2,3))
            batchStatistic[1] = torch.std(input, dim=(0,2,3))
            batchStatistic2[0] = torch.mean(target, dim=(0,2,3))
            batchStatistic2[1] = torch.std(target, dim=(0,2,3))
            for i in range(input.shape[1]):
                input[:,i,...] = (input[:,i,...]-batchStatistic[0,i])/batchStatistic[1,i]
            for i in range(input.shape[1]):
                target[:,i,...] = (target[:,i,...]-batchStatistic2[0,i])/batchStatistic2[1,i]
            #print(input.shape, target.shape)
            prediction = model(input)
            #np.save(f"{TEST_Path}/{test_set.image_filenames[index][:-4]}_Prediction.npy", torch.permute(prediction[0], (1,2,0)).numpy())
            #print(target.shape)
            #print(prediction.shape)
            mse = criterion_mse(prediction[:,channel,...], target[:,channel,...])
            mae = criterion_mae(prediction[:,channel,...], target[:,channel,...])
            div = criterion_div(prediction, target)
            batch_mse = mse.item()
            epoch_mse += batch_mse
            batch_mae = mae.item()
            epoch_mae += batch_mae
            batch_div = div.item()
            epoch_div += batch_div
            mse_list.append((test_set.image_filenames[index],batch_mse))
            psnr = 20 * log10((torch.max(target[:,channel,...]) - torch.min(target[:,channel,...]))/ np.sqrt(batch_mse))
            avg_psnr += psnr
            psnr_list.append((test_set.image_filenames[index], psnr))
            mae_list.append((test_set.image_filenames[index],batch_mae))
            div_list.append((test_set.image_filenames[index],batch_div))
            #print(f"{test_set.image_filenames[index]}, {batch_mse}, {psnr}")
    print("===> Avg. MSE: {:.4f}".format(epoch_mse / (len(testing_data_loader)//BATCH_SIZE)))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / (len(testing_data_loader)//BATCH_SIZE)))
    print(torch.max(target[:,channel,...]), torch.min(target[:,channel,...]))
    full_losses = (epoch_mse, avg_psnr, epoch_mae, epoch_div)
    avg_losses = np.array(full_losses)/(len(testing_data_loader)//BATCH_SIZE)
    list_losses = (mse_list, psnr_list, mae_list, div_list)
    #return epoch_mse, epoch_mse / len(testing_data_loader), mse_list, avg_psnr, avg_psnr / len(testing_data_loader), psnr_list
    return full_losses, avg_losses, list_losses

def log_seperate_epoch(metrics, metric_names, logging_path):
    save_path = Path(logging_path)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    #np.savetxt(f"{save_path}/test_mse.csv", test_mse)
    #np.savetxt(f"{save_path}/test_psnr.csv", test_psnr)
    for i in range(len(metrics)):
        pd.DataFrame(metrics[i]).to_csv(f"{save_path}/test_{metric_names[i]}.csv", index=False)

BATCH_SIZE = 1
#CHECKPOINT_PATH = "../Logs_from_the_cloud/Friday-24-11-2023/model_checkpoints/model_epoch_30.pth"
CHECKPOINT_PATH = "../Logs_from_the_cloud/Sunday-21-01-2024-4/model_checkpoints/model_epoch_60.pth"
SAVE_PATH = "test_logs/PresentationTests/"

if torch.cuda.is_available():
    print("Cuda is available")
    device = torch.device("cuda")
    checkpoint_dict = torch.load(CHECKPOINT_PATH)
else:
    device = torch.device("cpu")
    checkpoint_dict = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))

test_set = get_test_set()
testing_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)

model1 = Net(upscale_factor=8).to(device)
criterion_mae = nn.L1Loss()
criterion_mse = nn.MSELoss()
criterion_div = div_loss()
loss_names = ("MSE", "PSNR", "MAE", "DIV")
model1.load_state_dict(checkpoint_dict["model_state_dict"])
print(len(testing_data_loader))
model1_start = perf_counter()
testChannel = (0,1,2)
#test_mse1, test_avg_mse1, test_iter_mse1, test_psnr1, test_avg_psnr1, test_iter_psnr1 = test(model1, criterion_mse)
full_loss1, avg_loss1, list_loss1 = test(model1, testChannel)
model1_end = perf_counter()
print(f"Time taken for model 1 = {model1_end-model1_start}")

#M1_Path = join(SAVE_PATH, "NewModel-B128-E120-L0001-MSELoss-LastData", f"Channel{testChannel}")
#M1_Path = join(SAVE_PATH, "DSCMS-B128-E120-L0001-MSELoss-LastData_AddedMinInPSNR", f"Channel{testChannel}")
#M1_Path = join(SAVE_PATH, "OldModel-B128-E120-L0001-MSELoss-LastData", f"Channel{testChannel}")
M1_Path = join(SAVE_PATH, "DSCMS-B128-E120-L0001-MSELoss-Batch-Normalization2", f"Channel{testChannel}")
Path(M1_Path).mkdir(parents=True, exist_ok=True)
#log_seperate_epoch(test_iter_mse1, test_iter_psnr1, M1_Path)
log_seperate_epoch(list_loss1, loss_names, M1_Path)
with open(f"{M1_Path}/SingleResults.txt", "w") as f:
    #f.write(f"TOT_MSE = {test_mse1}\nTOT_PSNR = {test_psnr1}\nAVG_MSE = {test_avg_mse1}\nAVG_PSNR = {test_avg_psnr1}\nTime_Taken = {model1_end-model1_start}s")
    for i in range(len(loss_names)):
        f.write(f"TOT_{loss_names[i]} = {full_loss1[i]}\n")
    for i in range(len(loss_names)):
        f.write(f"AVG_{loss_names[i]} = {avg_loss1[i]}\n")

#exit()
model2 = lambda x : nn.functional.interpolate(x, (1536, 512), mode="bicubic")

model2_start = perf_counter()
#test_mse2, test_avg_mse2, test_iter_mse2, test_psnr2, test_avg_psnr2, test_iter_psnr2 = test(model2, criterion_mse)
full_loss2, avg_loss2, list_loss2 = test(model2, testChannel)
model2_end = perf_counter()
print(f"Time taken for model 2 = {model2_end-model2_start}")

M2_Path = join(SAVE_PATH, "Bicubic-Batch-Normalization2", f"Channel{testChannel}")
#log_seperate_epoch(test_iter_mse2, test_iter_psnr2, M2_Path)
log_seperate_epoch(list_loss2, loss_names, M2_Path)
with open(f"{M2_Path}/SingleResults.txt", "w") as f:
    #f.write(f"TOT_MSE = {test_mse2}\nTOT_PSNR = {test_psnr2}\nAVG_MSE = {test_avg_mse2}\nAVG_PSNR = {test_avg_psnr2}\nTime_Taken = {model2_end-model2_start}s")
    for i in range(len(loss_names)):
        f.write(f"TOT_{loss_names[i]} = {full_loss2[i]}\n")
    for i in range(len(loss_names)):
        f.write(f"AVG_{loss_names[i]} = {avg_loss2[i]}\n")
        
