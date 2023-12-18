from __future__ import print_function
import argparse
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, GaussianBlur, InterpolationMode
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from model import Net
#from multiscale_model import Net
#from ESPCN_model import Net
from math import log10
from torch.nn import MSELoss
from torch.nn.functional import interpolate

# Command I always use to run
# python super_resolve4.py data/all_data/test/data/T622-X1010-S32-A09.npy ../Logs_from_the_cloud/Thursday-30-11-2023/model_checkpoints/model_epoch_30.pth MSC_Model_First.png

# Training settings
parser = argparse.ArgumentParser(description='TorchVision Resize Demonstration')
parser.add_argument('input_image', type=str, help='input image to use')
parser.add_argument('output_filename', type=str, help='where to save the output image')
opt = parser.parse_args()

data = opt.input_image.split("/")[-1]
pwd = os.path.dirname(os.getcwd())
label_dir = os.path.join(os.getcwd(), "data", "all_data", "test", "labels", f"{data}")
# label = np.mean(np.load(label_dir), axis=2) 
label = np.load(label_dir)

max_Val = np.max(np.sum(label[...,1,:]**2, axis=2))
min_Val = np.min(np.sum(label[...,1,:]**2, axis=2))
print(min_Val, max_Val)

Fig, ax = plt.subplots(2, 3)

ax[0,2].imshow(np.sum(label[...,1,:]**2, axis=2), vmin=min_Val, vmax=max_Val)
ax[0,2].axis("off")
ax[0,2].set_title("DNS (reference)")

print(opt)
img = np.load(opt.input_image)
img = np.float32(img)

img_to_tensor = Compose([
        ToTensor(),
        GaussianBlur(9, 1),
        #Resize((48,16), antialias=True),
        #Resize((192, 64), antialias=False, interpolation=InterpolationMode.BILINEAR)
    ])

input = img_to_tensor(img).to("cpu")
input.unsqueeze_(0)
print(input.shape)

input = input[0]
input = torch.permute(input, (1,2,0))
input = input.data.numpy()

ax[0,0].imshow(np.sum(input**2, axis=2), vmin=min_Val, vmax=max_Val)
ax[0,0].set_title("Non-Resized")
ax[0,0].axis("off")

img_to_tensor1 = Compose([
        ToTensor(),
        GaussianBlur(9, 1),
        #Resize((48,16), antialias=True),
        Resize((192, 64), antialias=False, interpolation=InterpolationMode.BILINEAR)
    ])
out1 = img_to_tensor1(img).to("cpu")
out1.unsqueeze_(0)
print(input.shape)

out1 = out1[0]
out1 = torch.permute(out1, (1,2,0))
out_img_y1 = out1.data.numpy()

ax[0,1].imshow(np.sum(out_img_y1**2, axis=2), vmin=min_Val, vmax=max_Val)
ax[0,1].set_title("Bilinear")
ax[0,1].axis("off")

img_to_tensor2 = Compose([
        ToTensor(),
        GaussianBlur(9, 1),
        #Resize((48,16), antialias=True),
        Resize((192, 64), antialias=False, interpolation=InterpolationMode.BICUBIC)
    ])
out2 = img_to_tensor2(img).to("cpu")
out2.unsqueeze_(0)
print(input.shape)

out2 = out2[0]
out2 = torch.permute(out2, (1,2,0))
out_img_y2 = out2.data.numpy()

ax[1,0].imshow(np.sum(out_img_y2**2, axis=2), vmin=min_Val, vmax=max_Val)
ax[1,0].set_title("Bicubic")
ax[1,0].axis("off")

img_to_tensor3 = Compose([
        ToTensor(),
        GaussianBlur(9, 1),
        #Resize((48,16), antialias=True),
        Resize((192, 64), antialias=False, interpolation=InterpolationMode.NEAREST)
    ])
out3 = img_to_tensor3(img).to("cpu")
out3.unsqueeze_(0)
print(input.shape)

out3 = out3[0]
out3 = torch.permute(out3, (1,2,0))
out_img_y3 = out3.data.numpy()

ax[1,1].imshow(np.sum(out_img_y3**2, axis=2), vmin=min_Val, vmax=max_Val)
ax[1,1].set_title("Nearest")
ax[1,1].axis("off")

img_to_tensor4 = Compose([
        ToTensor(),
        GaussianBlur(9, 1),
        #Resize((48,16), antialias=True),
        Resize((192, 64), antialias=False, interpolation=InterpolationMode.NEAREST_EXACT)
    ])
out4 = img_to_tensor4(img).to("cpu")
out4.unsqueeze_(0)
print(input.shape)

out4 = out4[0]
out4 = torch.permute(out4, (1,2,0))
out_img_y4 = out4.data.numpy()

ax[1,2].imshow(np.sum(out_img_y4**2, axis=2), vmin=min_Val, vmax=max_Val)
ax[1,2].set_title("Nearest Exact")
ax[1,2].axis("off")

Fig.savefig(opt.output_filename, dpi=400, bbox_inches='tight')