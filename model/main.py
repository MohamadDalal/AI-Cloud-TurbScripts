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
from model import Net
from data import get_training_set, get_validation_set, get_test_set
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

def train(epoch):
    epoch_loss = 0
    loss_list = []
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))
        loss_list.append(loss.item())

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    return epoch_loss, epoch_loss / len(training_data_loader), loss_list


def validate():
    avg_psnr = 0
    psnr_list = []
    epoch_mse = 0
    mse_list = []
    with torch.no_grad():
        for batch in validation_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            epoch_mse += mse.item()
            mse_list.append(mse.item())
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
            psnr_list.append(psnr)
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(validation_data_loader)))
    return epoch_mse, epoch_mse / len(validation_data_loader), mse_list, avg_psnr, avg_psnr / len(validation_data_loader), psnr_list

def test():
    avg_psnr = 0
    psnr_list = []
    epoch_mse = 0
    mse_list = []
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            epoch_mse += mse.item()
            mse_list.append(mse.item())
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
            psnr_list.append(psnr)
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    return epoch_mse, epoch_mse / len(testing_data_loader), mse_list, avg_psnr, avg_psnr / len(testing_data_loader), psnr_list


def checkpoint(epoch):
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


def log_seperate_epoch(epoch, mse, validation_mse, validation_psnr):
    save_path = Path(join(logging_path, f"Epoch_{epoch}"))
    Path(save_path).mkdir(parents=train, exist_ok=True)
    np.savetxt(f"{save_path}/mse.csv", mse)
    np.savetxt(f"{save_path}/validation_mse.csv", validation_mse)
    np.savetxt(f"{save_path}/validation_psnr.csv", validation_psnr)

def log_all():
    np.savetxt(f"{logging_path}/mse.csv", all_mse)
    np.savetxt(f"{logging_path}/avg_mse.csv", all_avg_mse)
    np.savetxt(f"{logging_path}/validation_mse.csv", all_validation_mse)
    np.savetxt(f"{logging_path}/validation_avg_mse.csv", all_validation_avg_mse)
    np.savetxt(f"{logging_path}/validation_psnr.csv", all_validation_psnr)
    np.savetxt(f"{logging_path}/validation_avg_psnr.csv", all_validation_avg_psnr)


device = torch.device("cpu")
BATCH_SIZE = 16
EPOCHS = 30
START_EPOCH = 20
CHECKPOINT_PATH = f"model_checkpoints/model_epoch_{START_EPOCH}.pth"

print('===> Loading datasets')
train_set_dir = join(getcwd(), "data", "all_data", "train")
validation_set_dir = join(getcwd(), "data", "all_data", "validation")
test_set_dir = join(getcwd(), "data", "all_data", "test")

train_set = get_training_set()
validation_set = get_validation_set
test_set = get_test_set()

training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
validation_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
testing_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)

logging_path = join(getcwd(), "training_logs")
Path(logging_path).mkdir(parents=True, exist_ok=True)
all_mse, all_avg_mse, all_validation_mse, all_validation_avg_mse, all_validation_psnr, all_validation_avg_psnr = [], [], [], [], [] ,[]


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
model = Net(upscale_factor=32).to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
if START_EPOCH > 0:
    checkpoint_dict = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
    all_mse = [0 for _ in range(START_EPOCH)]
    all_avg_mse = [0 for _ in range(START_EPOCH)]
    all_validation_mse = [0 for _ in range(START_EPOCH)]
    all_validation_avg_mse = [0 for _ in range(START_EPOCH)]
    all_validation_psnr = [0 for _ in range(START_EPOCH)]
    all_validation_avg_psnr = [0 for _ in range(START_EPOCH)]

for epoch in range(START_EPOCH + 1, EPOCHS + 1):
    start_time = perf_counter()
    model.train()
    mse, avg_mse, iter_mse = train(epoch)
    model.eval()
    validation_mse, validation_avg_mse, validation_iter_mse, validation_psnr, validation_avg_psnr, validation_iter_psnr = validate()
    checkpoint(epoch)
    end_time = perf_counter()
    print(f"Epoch {epoch} took {end_time-start_time}s")
    all_mse.append(mse)
    all_avg_mse.append(avg_mse)
    all_validation_mse.append(validation_mse)
    all_validation_avg_mse.append(validation_avg_mse)
    all_validation_psnr.append(validation_psnr)
    all_validation_avg_psnr.append(validation_avg_psnr)
    log_seperate_epoch(epoch, iter_mse, validation_iter_mse, validation_iter_psnr)

for i in range(START_EPOCH):
    all_mse[i] = (all_mse[START_EPOCH])
    all_avg_mse[i] = (all_avg_mse[START_EPOCH])
    all_validation_mse[i] = (all_validation_mse[START_EPOCH])
    all_validation_avg_mse[i] = (all_validation_avg_mse[START_EPOCH])
    all_validation_psnr[i] = (all_validation_psnr[START_EPOCH])
    all_validation_avg_psnr[i] = (all_validation_avg_psnr[START_EPOCH])

log_all()