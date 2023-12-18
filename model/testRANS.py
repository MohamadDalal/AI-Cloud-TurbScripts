from __future__ import print_function
import argparse
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, GaussianBlur, InterpolationMode
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
#from old_model import Net
from multiscale_model import Net
#from model import Net


def is_array_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

img_to_tensor = Compose([
        ToTensor(),
        Resize((152,104), interpolation=InterpolationMode.BILINEAR),
    ])

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('model', type=str, help='model file to use')
parser.add_argument('suffix', type=str, help='suffix to put after the output\'s name')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

if torch.cuda.is_available() and opt.cuda:
    torch.device("cuda")
else:
    device = torch.device("cpu")

checkpoint_dict = torch.load(opt.model, map_location=torch.device("cpu"))
model = Net(upscale_factor=8).to(device)
model.load_state_dict(checkpoint_dict["model_state_dict"])
#model = torch.load(opt.model)
model.eval()
#img_to_tensor = ToTensor()

data_dir = "data_RANS"
image_filenames = [x for x in sorted(os.listdir(data_dir)) if is_array_file(x)]

for i, x in enumerate(image_filenames):
    if i>0:
        break
    img = np.load(f"{data_dir}/{x}")[...,4:]
    img = np.float32(img)
    print(img.shape)

    input = img_to_tensor(img).to(device)
    input.unsqueeze_(0)
    print(input.shape)

    out = model(input)
    #out = input
    out = out.cpu()
    out = out[0]
    #out_img_y = out[0].detach().numpy()
    out = torch.permute(out, (1,2,0))
    out_img_y = out.data.numpy()
    Fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(np.sum(out_img_y**2, axis=2))
    Fig.savefig(f"{data_dir}/{x[:-4]}_{opt.suffix}.png", dpi=400)
    ax.cla()
    plt.close()
