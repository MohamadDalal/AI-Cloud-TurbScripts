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

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('input_image', type=str, help='input image to use')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

plt.rcParams["font.family"] = "Liberation Serif"
#hfont = {'fontname':'Helvetica'}
#nsfont = {'fontname':'NimbusSans'}

print(opt)
img = np.load(opt.input_image)
img = np.float32(img)

device = torch.device("cpu")

DNS_to_Blur = Compose([
            ToTensor(),
            GaussianBlur(9, 1),
            #Resize((192,64), antialias=True),
])

DNS_to_RANS = Compose([
            ToTensor(),
            GaussianBlur(9, 1),
            Resize((192,64), antialias=True),
])

img_to_tensor = Compose([
            ToTensor(),
            #GaussianBlur(9, 1),
            #Resize((48,16), antialias=True),
        ])
label_to_tensor = Compose([ToTensor()])
input = img_to_tensor(img).to(device)
input.unsqueeze_(0)
input = input[0]
input = torch.permute(input, (1,2,0))
input = input.data.numpy()

blurred = DNS_to_Blur(img).to(device)
blurred.unsqueeze_(0)
blurred = blurred[0]
blurred = torch.permute(blurred, (1,2,0))
blurred = blurred.data.numpy()

resized = DNS_to_RANS(img).to(device)
resized.unsqueeze_(0)
resized = resized[0]
resized = torch.permute(resized, (1,2,0))
resized = resized.data.numpy()

data = opt.input_image.split("/")[-1]
pwd = os.path.dirname(os.getcwd())
label_dir = os.path.join(os.getcwd(), "data", "all_data", "test", "labels", f"{data}")
# label = np.mean(np.load(label_dir), axis=2) 
label = np.load(label_dir)

max_Val = np.max(np.sum(label[...,1,:]**2, axis=2))
min_Val = np.min(np.sum(label[...,1,:]**2, axis=2))
print(min_Val, max_Val)

fig, axes = plt.subplots(1,4, figsize=(10,10))
print(input.shape)
axes[0].imshow(np.sum(label[...,1,:]**2, axis=2), vmin=min_Val, vmax=max_Val)
axes[0].axis("off")
axes[0].set_title("DNS")
axes[1].imshow(np.sum(input**2, axis=2), vmin=min_Val, vmax=max_Val)
axes[1].axis("off")
axes[1].set_title("Averaged")
axes[2].imshow(np.sum(blurred**2, axis=2), vmin=min_Val, vmax=max_Val)
axes[2].axis("off")
axes[2].set_title("Blurred")
axes[3].imshow(np.sum(resized**2, axis=2), vmin=min_Val, vmax=max_Val)
axes[3].axis("off")
axes[3].set_title("Bilinear")
#pcm = axes[0].pcolormesh(np.random.random((20, 20)) * (0 + 1),
#                            cmap='viridis')
#fig.colorbar(pcm, ax=axes[:], shrink=0.5)
fig.savefig("FinalReportFigures/DNS_to_RANS_Figure/FigureNewFont.png", dpi=800, bbox_inches='tight')


Fig, ax = plt.subplots()
ax.imshow(np.sum(blurred**2, axis=2), vmin=min_Val, vmax=max_Val)
ax.axis("off")
#ax.set_title(Titles[i])
Fig.savefig(f"FinalReportFigures/DNS_to_RANS_Figure/Blurred.png", dpi=600, bbox_inches='tight')

Fig, ax = plt.subplots()
ax.imshow(np.sum(resized**2, axis=2), vmin=min_Val, vmax=max_Val)
ax.axis("off")
#ax.set_title(Titles[i])
Fig.savefig(f"FinalReportFigures/DNS_to_RANS_Figure/RANS.png", dpi=600, bbox_inches='tight')

Fig, ax = plt.subplots()
ax.imshow(np.sum(input**2, axis=2), vmin=min_Val, vmax=max_Val)
ax.axis("off")
#ax.set_title(Titles[i])
Fig.savefig(f"FinalReportFigures/DNS_to_RANS_Figure/Averaged.png", dpi=600, bbox_inches='tight')

Fig, ax = plt.subplots()
ax.imshow(np.sum(label[...,1,:]**2, axis=2), vmin=min_Val, vmax=max_Val)
ax.axis("off")
#ax.set_title(Titles[i])
Fig.savefig(f"FinalReportFigures/DNS_to_RANS_Figure/DNS.png", dpi=600, bbox_inches='tight')


"""
fig.text(0.21, 0.1, "MSE", wrap=True, horizontalalignment='center', fontsize=12)
#text = "{:.5f}".format(bicubic_mse[bicubic_mse.iloc[:,0] == data].values[0][1])
text = "{:.5f}".format(bicubic_mse)
#text = round(bicubic_mse[bicubic_mse.iloc[:,0] == data].values[0][1], 5)
#print(text)
fig.text(0.41, 0.1, text, wrap=True, horizontalalignment='center', fontsize=12)
#text = "{:.5f}".format(model_mse[model_mse.iloc[:,0] == data].values[0][1])
text = "{:.5f}".format(model_mse)
fig.text(0.61, 0.1, text, wrap=True, horizontalalignment='center', fontsize=12)
fig.text(0.21, 0.05, "PSNR", wrap=True, horizontalalignment='center', fontsize=12)
#text = "{:.5f}".format(bicubic_psnr[bicubic_psnr.iloc[:,0] == data].values[0][1])
text = "{:.5f}".format(bicubic_psnr)
fig.text(0.41, 0.05, text, wrap=True, horizontalalignment='center', fontsize=12)
#text = "{:.5f}".format(model_psnr[model_psnr.iloc[:,0] == data].values[0][1])
text = "{:.5f}".format(model_psnr)
fig.text(0.61, 0.05, text, wrap=True, horizontalalignment='center', fontsize=12)
"""


# out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
# out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
# out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

# out_img.save(opt.output_filename)
# print('output image saved to ', opt.output_filename)