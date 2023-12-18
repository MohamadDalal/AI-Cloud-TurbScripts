from __future__ import print_function
import argparse
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, GaussianBlur
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from model import Net
from multiscale_model import Net as DSCMS_Net
#from ESPCN_model import Net
#from old_model import Net
from math import log10
from torch.nn import MSELoss, L1Loss
from torch.nn.functional import interpolate
from os.path import join
from Custom_loss import div_loss


# Command I always use to run
# python super_resolve4.py data49x18/all_data/test/data/T622-X1010-S32-A09.npy ../Logs_from_the_cloud/Thursday-30-11-2023/model_checkpoints/model_epoch_30.pth MSC_Model_First.png



plt.rcParams["font.family"] = "Liberation Serif"

#plt.rcParams["font.family"] = "Helvetica"
#hfont = {'fontname':'Helvetica'}

img = np.load("data_RANS/X0.0.npy")[...,4:]
img = np.float32(img)

device = torch.device("cpu")

input_to_tensor = Compose([
            ToTensor(),
            GaussianBlur(9, 1),
            Resize((152,104), antialias=True),
])

img_to_tensor = Compose([
            ToTensor(),
            #GaussianBlur(9, 1),
            #Resize((48,16), antialias=True),
        ])
label_to_tensor = Compose([ToTensor()])
input = input_to_tensor(img).to(device)
input.unsqueeze_(0)
print(input.shape)


critereon_MSE = MSELoss()
critereon_MAE = L1Loss()
critereon_DIV = div_loss()

MODEL_PATH = "../Logs_from_the_cloud/"
CHECKPOINT_PATH = "model_checkpoints/model_epoch_120.pth"

Checkpoint = join(MODEL_PATH, "Wednesday-06-12-2023-2/", CHECKPOINT_PATH)
Title = "DSC\MS with\nPixle Shuffle"



out2 = interpolate(input, size=(1216,832), mode="bicubic")
out2 = out2.cpu()
out2 = out2[0]
#out_img_y = out[0].detach().numpy()
out2 = torch.permute(out2, (1,2,0))
out_img_y2 = out2.data.numpy()



checkpoint_dict = torch.load(Checkpoint, map_location=torch.device("cpu"))
model = DSCMS_Net(upscale_factor=8).to(device)
model.load_state_dict(checkpoint_dict["model_state_dict"])
#model = torch.load(opt.model)
model.eval()
#img_to_tensor = ToTensor()

out = model(input)
out = out.cpu()
out = out[0]
#print(out.shape)
out = torch.permute(out, (1,2,0))
#print(out[None,...].shape)
#print(label_to_tensor(label[...,1,:]).shape)
out_img_y = out.data.numpy()

    
max_Val = np.max(np.sum(input[0].numpy()**2, axis=0))
min_Val = np.min(np.sum(input[0].numpy()**2, axis=0))
print(min_Val, max_Val)
colorbarTicks = np.linspace(min_Val, max_Val, 10, endpoint=True)

fig, axes = plt.subplots(1,3, figsize=(10,5))
pcm = axes[0].imshow(np.sum(input[0].numpy()**2, axis=0), vmin=min_Val, vmax=max_Val)
axes[0].axis("off")
axes[0].set_title("Input image")

axes[2].imshow(np.sum(out_img_y**2, axis=2), vmin=min_Val, vmax=max_Val)
axes[2].axis("off")
axes[2].set_title(Title)

axes[1].imshow(np.sum(out_img_y2**2, axis=2), vmin=min_Val, vmax=max_Val)
axes[1].axis("off")
axes[1].set_title("Bicubic interpolation")
print(pcm)
fig.colorbar(pcm, ax=axes[:], shrink=0.7)
fig.savefig("FinalReportFigures/Figure_DSCMS_Real_RANS/DSCMS_Real_RANS.png", dpi=800, bbox_inches='tight')

Fig, ax = plt.subplots()
ax.imshow(np.sum(input[0].numpy()**2, axis=0), vmin=min_Val, vmax=max_Val)
ax.axis("off")
#ax.set_title(Titles[i])
Fig.savefig(f"FinalReportFigures/Figure_DSCMS_Real_RANS/input.png", dpi=600, bbox_inches='tight')

Fig, ax = plt.subplots()
ax.imshow(np.sum(out_img_y**2, axis=2), vmin=min_Val, vmax=max_Val)
ax.axis("off")
#ax.set_title(Titles[i])
Fig.savefig(f"FinalReportFigures/Figure_DSCMS_Real_RANS/{Title}.png", dpi=600, bbox_inches='tight')


Fig, ax = plt.subplots()
ax.imshow(np.sum(out_img_y2**2, axis=2), vmin=min_Val, vmax=max_Val)
ax.axis("off")
#ax.set_title(Titles[i])
Fig.savefig(f"FinalReportFigures/Figure_DSCMS_Real_RANS/Bicubic.png", dpi=600, bbox_inches='tight')
