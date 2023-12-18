import torch
import torch.nn as nn
import torch.nn.init as init
#from torchinfo import summary


class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()
        self.upscale_factor = upscale_factor
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, (13, 13), (1, 1), (6, 6))
        self.conv2 = nn.Conv2d(64, 64, (9, 9), (1, 1), (4, 4))
        self.conv3 = nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2))
        self.conv4 = nn.Conv2d(32, 3 * self.upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)

        self._initialize_weights()


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        # x = self.conv4(x)
        # x1 = self.pixel_shuffle(x[:,:self.upscale_factor ** 2, ...])
        # x2 = self.pixel_shuffle(x[:,self.upscale_factor ** 2:2 * self.upscale_factor ** 2,...])
        # x3 = self.pixel_shuffle(x[:,2 * self.upscale_factor ** 2:,...])
        # x = torch.cat((x1,x2,x3), dim=1)
        return x


    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


if __name__ == "__main__":
    # Create an instance of the model
    model = Net(upscale_factor=8)

    # Print the model architecture
    #print(summary(model, (1,3,192,64)))