import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, upscale_factor):
        #ESPCN Model class

        super(Net, self).__init__()
        self.upscale_factor = upscale_factor
        #self.relu = nn.ReLU()

        # As per paper, 3 conv layers in backbone, adding padding is optional, not mentioned to use in paper
        # SRCNN paper does not recommend using padding, padding here just helps to visualize the scaled up output image
        # Extract input image feature maps
        self.feature_map_layer = nn.Sequential(
            # (f1,n1) = (5, 64)
            nn.Conv2d(in_channels=3, kernel_size=(5, 5), out_channels=64, padding=(2, 2)),
            # Using "Tanh" activation instead of "ReLU"
            nn.Tanh(),
            # (f2,n2) = (3, 32)
            nn.Conv2d(in_channels=64, kernel_size=(3, 3), out_channels=32, padding=(1, 1)),
            # Using "Tanh" activation instead of "ReLU"
            nn.Tanh()
        )

        self.sub_pixel_layer = nn.Sequential(
            # f3 = 3, # output shape: H x W x (C x r**2)
            nn.Conv2d(in_channels=32, kernel_size=(3, 3), out_channels=3 * (upscale_factor ** 2), padding=(1, 1)),
            # Sub-Pixel Convolution Layer - PixelShuffle
            # rearranges: H x W x (C x r**2) => rH x rW x C
            nn.PixelShuffle(upscale_factor=upscale_factor)
        )

    def forward(self, x):
        """

        :param x: input image
        :return: model output
        """

        # inputs: H x W x C
        x = self.feature_map_layer(x)
        # output: rH x rW x C
        # r: scale_factor
        out = self.sub_pixel_layer(x)

        return out

