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
#from multiscale_model import Net
#from ESPCN_model import Net
#from old_model import Net
from math import log10
from torch.nn import MSELoss, L1Loss
from torch.nn.functional import interpolate
from os.path import join
from Custom_loss import div_loss


# Command I always use to run
# python super_resolve4.py data/all_data/test/data/T622-X1010-S32-A09.npy ../Logs_from_the_cloud/Thursday-30-11-2023/model_checkpoints/model_epoch_30.pth MSC_Model_First.png

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('input_image', type=str, help='input image to use')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

print(opt)
img = np.load(opt.input_image)
img = np.float32(img)

if torch.cuda.is_available() and opt.cuda:
    torch.device("cuda")
else:
    device = torch.device("cpu")

img_to_tensor = Compose([
            ToTensor(),
            #GaussianBlur(9, 1),
            #Resize((48,16), antialias=True),
        ])
label_to_tensor = Compose([ToTensor()])
input = img_to_tensor(img).to(device)
input.unsqueeze_(0)
print(input.shape)

data = opt.input_image.split("/")[-1]
pwd = os.path.dirname(os.getcwd())
label_dir = os.path.join(os.getcwd(), "data", "all_data", "test", "labels", f"{data}")
# label = np.mean(np.load(label_dir), axis=2) 
label = np.load(label_dir)

critereon_MSE = MSELoss()
critereon_MAE = L1Loss()
critereon_DIV = div_loss()

MODEL_PATH = "../Logs_from_the_cloud/"
CHECKPOINT_PATH = "model_checkpoints/model_epoch_30.pth"

Checkpoints = [join(MODEL_PATH, "Wednesday-29-11-2023/", CHECKPOINT_PATH),
               join(MODEL_PATH, "Wednesday-29-11-2023-2/", CHECKPOINT_PATH), 
               join(MODEL_PATH, "Wednesday-29-11-2023-3/", CHECKPOINT_PATH)]
Titles = ["MSE Output", "MAE Output", 'DIV Output']

out_img_ys = []
MSEs = []
PSNRs = []
MAEs = []
DIVs = []


for i in range(3):
    checkpoint_dict = torch.load(Checkpoints[i], map_location=torch.device("cpu"))
    model = Net(upscale_factor=8).to(device)
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
    out_img_ys.append(out.data.numpy())
    model_mse = critereon_MSE(torch.permute(out, (2,0,1)), label_to_tensor(label[...,1,:]))
    model_mae = critereon_MAE(torch.permute(out, (2,0,1)), label_to_tensor(label[...,1,:]))
    model_div = critereon_DIV(torch.permute(out, (2,0,1))[None,...], label_to_tensor(label[...,1,:])[None,...])
    MSEs.append(model_mse.item())
    PSNRs.append(20 * log10(np.max(label[...,1,:]) / np.sqrt(MSEs[i])))
    MAEs.append(model_mae.item())
    DIVs.append(model_div.item())

    
max_Val = np.max(np.sum(label[...,1,:]**2, axis=2))
min_Val = np.min(np.sum(label[...,1,:]**2, axis=2))
print(min_Val, max_Val)

fig, axes = plt.subplots(1,5, figsize=(10,5))
print(input.numpy().shape)
axes[0].imshow(np.sum(input[0].numpy()**2, axis=0), vmin=min_Val, vmax=max_Val)
axes[0].axis("off")
axes[0].set_title("Input image")
for i in range(1,4):
    axes[i].imshow(np.sum(out_img_ys[i-1]**2, axis=2), vmin=min_Val, vmax=max_Val)
    axes[i].axis("off")
    axes[i].set_title(Titles[i-1])
axes[4].imshow(np.sum(label[...,1,:]**2, axis=2), vmin=min_Val, vmax=max_Val)
axes[4].axis("off")
axes[4].set_title("DNS (reference)")
fig.savefig("FinalReportFigures/NewModel_DifferentLosses/Figure.png", dpi=400, bbox_inches='tight')
with open("FinalReportFigures/NewModel_DifferentLosses/Stats.txt", "w") as f:
    f.write("MSE\n")
    for i in range(3):
        f.write(f"\t{Titles[i]}: {MSEs[i]}\n")
    f.write("PSNR\n")
    for i in range(3):
        f.write(f"\t{Titles[i]}: {PSNRs[i]}\n")
    f.write("MAE\n")
    for i in range(3):
        f.write(f"\t{Titles[i]}: {MAEs[i]}\n")
    f.write("DIV\n")
    for i in range(3):
        f.write(f"\t{Titles[i]}: {DIVs[i]}\n")


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