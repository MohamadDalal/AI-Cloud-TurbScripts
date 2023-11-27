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
from torch.nn.functional import interpolate


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('input_image', type=str, help='input image to use')
parser.add_argument('model', type=str, help='model file to use')
parser.add_argument('output_filename', type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

print(opt)
img = np.load(opt.input_image)
img = np.float32(img)

if torch.cuda.is_available() and opt.cuda:
    torch.device("cuda")
else:
    device = torch.device("cpu")

checkpoint_dict = torch.load(opt.model, map_location=torch.device("cpu"))
model = Net(upscale_factor=32).to(device)
model.load_state_dict(checkpoint_dict["model_state_dict"])
#model = torch.load(opt.model)
model.eval()
#img_to_tensor = ToTensor()
img_to_tensor = Compose([
        ToTensor(),
        GaussianBlur(9, 1),
        Resize((48,16), antialias=True),
    ])
input = img_to_tensor(img).to(device)
input.unsqueeze_(0)
print(input.shape)

out = model(input)
out = out.cpu()
out = out[0]
#out_img_y = out[0].detach().numpy()
out = torch.permute(out, (1,2,0))
out_img_y = out.data.numpy()
# out_img_y *= 255.0
# out_img_y = out_img_y.clip(0, 255)
# out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

out2 = interpolate(input, size=(1536,512), mode="bicubic")
out2 = out2.cpu()
out2 = out2[0]
#out_img_y = out[0].detach().numpy()
out2 = torch.permute(out2, (1,2,0))
out2_img_y = out2.data.numpy()

data = opt.input_image.split("/")[-1]
pwd = os.path.dirname(os.getcwd())
label_dir = os.path.join(os.getcwd(), "data", "all_data", "test", "labels", f"{data}")
# label = np.mean(np.load(label_dir), axis=2) 
label = np.load(label_dir)


model_mse = pd.read_csv(f"test_logs/M1-B128-E30-L001-FixedResize-FixedGaussian-FixedDataloader/test_mse.csv")
model_psnr = pd.read_csv(f"test_logs/M1-B128-E30-L001-FixedResize-FixedGaussian-FixedDataloader/test_psnr.csv")
bicubic_mse = pd.read_csv(f"test_logs/Bicubic-FixedDataloader/test_mse.csv")
bicubic_psnr = pd.read_csv(f"test_logs/Bicubic-FixedDataloader/test_psnr.csv")
#print("{:.5f}".format(model_mse[model_mse.iloc[:,0] == data].values[0][1]))
#exit()

fig, axes = plt.subplots(1,4)
print(input.numpy().shape)
axes[0].imshow(np.mean(input[0].numpy(), axis=0))
axes[0].axis("off")
axes[0].set_title("Input image")
axes[1].imshow(np.mean(out2_img_y, axis=2))
axes[1].axis("off")
axes[1].set_title("Bicubic output")
axes[2].imshow(np.mean(out_img_y, axis=2))
axes[2].axis("off")
axes[2].set_title("Model output")
axes[3].imshow(np.mean(label[...,1,:], axis=2))
axes[3].axis("off")
axes[3].set_title("DNS (reference)")
# Automate metrics retrieval if needed
fig.text(0.21, 0.1, "MSE", wrap=True, horizontalalignment='center', fontsize=12)
text = "{:.5f}".format(bicubic_mse[bicubic_mse.iloc[:,0] == data].values[0][1])
#text = round(bicubic_mse[bicubic_mse.iloc[:,0] == data].values[0][1], 5)
#print(text)
fig.text(0.41, 0.1, text, wrap=True, horizontalalignment='center', fontsize=12)
text = "{:.5f}".format(model_mse[model_mse.iloc[:,0] == data].values[0][1])
fig.text(0.61, 0.1, text, wrap=True, horizontalalignment='center', fontsize=12)
fig.text(0.21, 0.05, "PSNR", wrap=True, horizontalalignment='center', fontsize=12)
text = "{:.5f}".format(bicubic_psnr[bicubic_psnr.iloc[:,0] == data].values[0][1])
fig.text(0.41, 0.05, text, wrap=True, horizontalalignment='center', fontsize=12)
text = "{:.5f}".format(model_psnr[model_psnr.iloc[:,0] == data].values[0][1])
fig.text(0.61, 0.05, text, wrap=True, horizontalalignment='center', fontsize=12)
fig.savefig(opt.output_filename, dpi=400, bbox_inches='tight')


# out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
# out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
# out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

# out_img.save(opt.output_filename)
# print('output image saved to ', opt.output_filename)