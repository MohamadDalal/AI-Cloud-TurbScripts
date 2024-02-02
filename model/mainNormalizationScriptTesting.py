from __future__ import print_function
import argparse
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from os.path import join
from os import getcwd
from pathlib import Path
#from model import Net
#from model2 import Net
from multiscale_model import Net
from data import get_training_set, get_validation_set, get_test_set
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from Custom_loss import div_loss

def train(epoch):
    epoch_loss = 0
    loss_list = []
    batchStatistic = torch.zeros((2,3))
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)
        #print(torch.mean(input, dim=(0,2,3)), torch.var(input, dim=(0,2,3)))
        #for i in range(input.shape[1]):
        #    input[:,i,...] = (input[:,i,...]-normalizationStatistics[0,i])/normalizationStatistics[1,i]
        #print(torch.mean(input, dim=(0,2,3)), torch.var(input, dim=(0,2,3)))
        #for i in range(input.shape[1]):
        #    input[:,i,...] = normalizationStatistics[1,i]*input[:,i,...]+normalizationStatistics[0,i]
        #print(torch.mean(input, dim=(0,2,3)), torch.var(input, dim=(0,2,3)))
        #print()
        batchStatistic[0] = torch.mean(input, dim=(0,2,3))
        batchStatistic[1] = torch.std(input, dim=(0,2,3))
        print(torch.mean(input, dim=(0,2,3)), torch.var(input, dim=(0,2,3)))
        for i in range(input.shape[1]):
            input[:,i,...] = (input[:,i,...]-batchStatistic[0,i])/batchStatistic[1,i]
        print(torch.mean(input, dim=(0,2,3)), torch.var(input, dim=(0,2,3)))
        for i in range(input.shape[1]):
            input[:,i,...] = batchStatistic[1,i]*input[:,i,...]+batchStatistic[0,i]
        print(torch.mean(input, dim=(0,2,3)), torch.var(input, dim=(0,2,3)))
        print()
        
        #optimizer.zero_grad()
        #output = model(input)
        #print(torch.mean(output, dim=(0,2,3)), torch.var(output, dim=(0,2,3)))
        #for i in range(output.shape[1]):
        #    output[:,i,...] = normalizationStatistics[1,i]*output[:,i,...]+normalizationStatistics[0,i]
        #print(torch.mean(output, dim=(0,2,3)), torch.var(output, dim=(0,2,3)))
        #loss = criterion(output, target)
        #MSE = criterion_mse(output, target)
        #DIV = criterion_div(output, target)
        #loss = WEIGHT*MSE + (1-WEIGHT)*DIV
        #batch_mse = loss.item()
        #epoch_loss += batch_mse
        #loss.backward()
        #optimizer.step()

        #print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), batch_mse))
        #loss_list.append(batch_mse)

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    return epoch_loss, epoch_loss / len(training_data_loader), loss_list


def validate():
    avg_psnr = 0
    psnr_list = []
    epoch_loss = 0
    loss_list = []
    with torch.no_grad():
        for batch in validation_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            loss = criterion(prediction, target)
            #MSE = criterion_mse(prediction, target)
            #DIV = criterion_div(prediction, target)
            #loss = WEIGHT*MSE + (1-WEIGHT)*DIV
            batch_loss = loss.item()
            epoch_loss += batch_loss
            loss_list.append(batch_loss)
            psnr = 10 * log10(1 / batch_loss)
            avg_psnr += psnr
            psnr_list.append(psnr)
    print("===> Validation {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    #print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(validation_data_loader)))
    return epoch_loss, epoch_loss / len(validation_data_loader), loss_list, avg_psnr, avg_psnr / len(validation_data_loader), psnr_list


def checkpoint(epoch):
    #model_out_dir = join(getcwd(), f"model_checkpoints_mixedLoss{WEIGHT}")
    model_out_dir = join(getcwd(), "model_checkpoints")
    Path(model_out_dir).mkdir(parents=True, exist_ok=True)
    model_out_path = "{}/model_epoch_{}.pth".format(model_out_dir, epoch)
    #torch.save(model, model_out_path)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
        }, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def log_seperate_epoch(epoch, loss, validation_loss, validation_psnr):
    save_path = Path(join(logging_path, f"Epoch_{epoch}"))
    Path(save_path).mkdir(parents=True, exist_ok=True)
    np.savetxt(f"{save_path}/loss.csv", loss)
    np.savetxt(f"{save_path}/validation_loss.csv", validation_loss)
    np.savetxt(f"{save_path}/validation_psnr.csv", validation_psnr)

def log_all():
    np.savetxt(f"{logging_path}/loss.csv", all_loss)
    np.savetxt(f"{logging_path}/avg_loss.csv", all_avg_loss)
    np.savetxt(f"{logging_path}/validation_loss.csv", all_validation_loss)
    np.savetxt(f"{logging_path}/validation_avg_loss.csv", all_validation_avg_loss)
    np.savetxt(f"{logging_path}/validation_psnr.csv", all_validation_psnr)
    np.savetxt(f"{logging_path}/validation_avg_psnr.csv", all_validation_avg_psnr)

if torch.cuda.is_available():
    print("Cuda is available")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

BATCH_SIZE = 100
EPOCHS = 1
START_EPOCH = 0
CHECKPOINT_PATH = f"model_checkpoints/model_epoch_{START_EPOCH}.pth"
WEIGHT = 0.99

normalizationStatistics = np.load("calcMeanVar/separatedStatistics2.npy")
normalizationStatistics[1,:] = np.sqrt(normalizationStatistics[1,:])

print('===> Loading datasets')
train_set_dir = join(getcwd(), "data", "all_data", "train")
validation_set_dir = join(getcwd(), "data", "all_data", "validation")
test_set_dir = join(getcwd(), "data", "all_data", "test")

train_set = get_test_set()
validation_set = get_test_set()
test_set = get_test_set()

training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
validation_data_loader = DataLoader(dataset=validation_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
testing_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)

#logging_path = join(getcwd(), f"training_logs_mixedLoss{WEIGHT}")
logging_path = join(getcwd(), "training_logs")
Path(logging_path).mkdir(parents=True, exist_ok=True)
all_loss, all_avg_loss, all_validation_loss, all_validation_avg_loss, all_validation_psnr, all_validation_avg_psnr = [], [], [], [], [] ,[]


print('===> Building model')
"""
if START_EPOCH > 0:
    model = torch.load(CHECKPOINT_PATH)
    all_mse = [0 for _ in range(START_EPOCH)]
    all_avg_mse = [0 for _ in range(START_EPOCH)]
    all_validation_mse = [0 for _ in range(START_EPOCH)]
    all_validation_avg_mse = [0 for _ in range(START_EPOCH)]
    all_validation_psnr = [0 for _ in range(START_EPOCH)]
    all_validation_avg_psnr = [0 for _ in range(START_EPOCH)]
else:
    model = Net(upscale_factor=32).to(device)
"""
model = Net(upscale_factor=8).to(device)
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
criterion_div = div_loss()
criterion = criterion_mse

optimizer = optim.Adam(model.parameters(), lr=0.0001)
if START_EPOCH > 0:
    checkpoint_dict = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
    all_loss = [0 for _ in range(START_EPOCH)]
    all_avg_loss = [0 for _ in range(START_EPOCH)]
    all_validation_loss = [0 for _ in range(START_EPOCH)]
    all_validation_avg_loss = [0 for _ in range(START_EPOCH)]
    all_validation_psnr = [0 for _ in range(START_EPOCH)]
    all_validation_avg_psnr = [0 for _ in range(START_EPOCH)]

for epoch in range(START_EPOCH + 1, EPOCHS + 1):
    start_time = perf_counter()
    model.train()
    mse, avg_mse, iter_mse = train(epoch)
    model.eval()
    #validation_loss, validation_avg_loss, validation_iter_loss, validation_psnr, validation_avg_psnr, validation_iter_psnr = validate()
    #checkpoint(epoch)
    end_time = perf_counter()
    print(f"Epoch {epoch} took {end_time-start_time}s")
    all_loss.append(mse)
    all_avg_loss.append(avg_mse)
    #all_validation_loss.append(validation_loss)
    #all_validation_avg_loss.append(validation_avg_loss)
    #all_validation_psnr.append(validation_psnr)
    #all_validation_avg_psnr.append(validation_avg_psnr)
    #log_seperate_epoch(epoch, iter_mse, validation_iter_loss, validation_iter_psnr)

for i in range(START_EPOCH):
    all_loss[i] = (all_loss[START_EPOCH])
    all_avg_loss[i] = (all_avg_loss[START_EPOCH])
    all_validation_loss[i] = (all_validation_loss[START_EPOCH])
    all_validation_avg_loss[i] = (all_validation_avg_loss[START_EPOCH])
    all_validation_psnr[i] = (all_validation_psnr[START_EPOCH])
    all_validation_avg_psnr[i] = (all_validation_avg_psnr[START_EPOCH])

#log_all()
