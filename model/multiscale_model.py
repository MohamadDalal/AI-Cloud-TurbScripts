import torch
import torch.nn as nn
import torch.nn.init as init
from torchinfo import summary

"""First, we extend the CNN model by introducing compression and
skipped connections, as shown in the red box of figure 3(c). In super-resolution analysis,
data compression (triangular operations) increases the robustness against translation and
rotation of the data elements (Le et al. 2010). The use of skipped connections (red arrows)
enhances the CNN prediction by removing issues related to the convergence of weights (He
et al. 2016) which is known to be a problem with deep CNNs. We also introduce the multi-
scale model by Du et al. (2018) that captures the small-scale structures in the data. This
multi-scale model is shown in the yellow box of figure 3(c) and is comprised of a number
of CNN filters with different lengths to span a range of scales. The extended super-
resolution approach combines the DSC and MS models, and is referred to as the hybrid
Downsampled Skip-Connection/Multi-Scale (DSC/MS) mode """




class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()
        self.upscale_factor = upscale_factor
        self.relu = nn.ReLU()


        # skip-connections
        # 1
        self.pool1 = nn.MaxPool2d(8, stride=8)
        self.convs1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.convs2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')

        # 2
        self.pool2 = nn.MaxPool2d(4, stride=4)
        self.convs3 = nn.Conv2d(35, 32, kernel_size=3, padding=1)
        self.convs4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        # 3
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.convs5 = nn.Conv2d(35, 32, kernel_size=3, padding=1)
        self.convs6 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        #pix-shuff
        # self.convs7 = nn.Conv2d(32, 3 * self.upscale_factor ** 2, (3,3), (1,1), (1,1))
        # self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)

        # 4
        self.convs8 = nn.Conv2d(35, 32, kernel_size=3, padding=1)
        self.convs9 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # first layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2) #padding = (kernelsize-1)/2 to ensure same resolution as input
        self.conv2 = nn.Conv2d(16, 8, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        

        # second layer
        self.conv4 = nn.Conv2d(3, 16, kernel_size=9, padding=4)
        self.conv5 = nn.Conv2d(16, 8, kernel_size=9, padding=4)
        self.conv6 = nn.Conv2d(8, 8, kernel_size=9, padding=4)

        # third layer
        self.conv7 = nn.Conv2d(3, 16, kernel_size=13, padding=6)
        self.conv8 = nn.Conv2d(16, 8, kernel_size=13, padding=6)
        self.conv9 = nn.Conv2d(8, 8, kernel_size=13, padding=6)

        # fourth layer
        self.conv10 = nn.Conv2d(27, 8, kernel_size=7, padding=3)
        self.conv11 = nn.Conv2d(8, 3, kernel_size=5, padding=2)

        # final layer/sub-pixel convolutional layer
        self.conv12 = nn.Conv2d(35, 3 * self.upscale_factor ** 2, (3, 3), (1, 1), (1, 1) )
        self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)


        self._initialize_weights()

    def forward(self, x):

        input_ = x
        down1 = self.pool1(input_)
        x1s = self.relu(self.convs1(down1))
        x1s = self.relu(self.convs2(x1s))
        x1s = self.upsample1(x1s)

        down2 = self.pool2(input_)
        x2s = torch.cat((x1s, down2), dim=1)
        x2s = self.relu(self.convs3(x2s))
        x2s = self.relu(self.convs4(x2s))
        x2s = self.upsample2(x2s)

        down3 = self.pool3(input_)
        x3s = torch.cat((x2s, down3), dim=1)
        x3s = self.relu(self.convs5(x3s))
        x3s = self.relu(self.convs6(x3s))
        x3s = self.upsample3(x3s)

        x4s = torch.cat((x3s, input_), dim=1)
        x4s = self.relu(self.convs8(x4s))
        x4s = self.relu(self.convs9(x4s))

        # first layer
        x1_1 = self.relu(self.conv1(input_))
        x1_2 = self.relu(self.conv2(x1_1))
        x1_3 = self.relu(self.conv3(x1_2))

        # second layer
        x2_1 = self.relu(self.conv4(input_))
        x2_2 = self.relu(self.conv5(x2_1))
        x2_3 = self.relu(self.conv6(x2_2))

        # third layer
        x3_1 = self.relu(self.conv7(x))
        x3_2 = self.relu(self.conv8(x3_1))
        x3_3 = self.relu(self.conv9(x3_2))

        # fourth layer
        x4_1 = torch.cat((x1_3, x2_3, x3_3, input_), dim=1)
        x4_2 = self.relu(self.conv10(x4_1))
        x4_3 = self.relu(self.conv11(x4_2))

        # pixel shuffle
        x_final = torch.cat((x4s, x4_3), dim=1) 
        x_final = self.pixel_shuffle(self.conv12(x_final))

        return x_final
    
    
    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv5.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv6.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv7.weight)

"""
    def __init__(self, upscale_factor):
        super(Net, self).__init__()
        self.upscale_factor = upscale_factor
        self.relu = nn.ReLU()

        # first layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2) #padding = (kernelsize-1)/2 to ensure same resolution as input
        self.conv2 = nn.Conv2d(16, 16, kernel_size=9, padding=4)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=13, padding=6)
        

        # second layer
        self.conv4 = nn.Conv2d(16, 8, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(8, 8, kernel_size=9, padding=4)
        self.conv6 = nn.Conv2d(8, 8, kernel_size=13, padding=6)


        #sub-pixel convolutional layer
        self.conv7 = nn.Conv2d(8, 3 * self.upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        # First layer
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Second layer
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))

        # Third layer + Pixel shuffle 
        x = self.relu(self.conv7(x))
        x = self.pixel_shuffle(self.conv6(x))

        return x

    def _initialize_weights(self):
        # Initialize the weights as before
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv5.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv6.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv7.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv8.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv9.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv10.weight)

"""

if __name__ == "__main__":
    # Create an instance of the model
    model = Net(upscale_factor=8)

    # Print the model architecture
    print(summary(model, (1,3,192,64)))
