import torch
import torch.nn as nn
import torch.nn.init as init

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

        # first layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=13, stride=1, padding=6) #padding = (kernelsize-1)/2 to ensure same resolution as input
        self.conv2 = nn.Conv2d(16, 16, kernel_size=13, stride=1, padding=6)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=13, stride=1, padding=6)
        

        # second layer
        self.conv4 = nn.Conv2d(16, 8, kernel_size=9, stride=1, padding=4)
        self.conv5 = nn.Conv2d(8, 8, kernel_size=9, stride=1, padding=4)
        self.conv6 = nn.Conv2d(8, 8, kernel_size=9, stride=1, padding=4)

        # third layer
        self.conv7 = nn.Conv2d(8,8,kernel_size=5,stride=1,padding=2)
        self.conv8 = nn.Conv2d(8,8,kernel_size=5,stride=1,padding=2)
        self.conv9 = nn.Conv2d(8,8,kernel_size=5,stride=1,padding=2)

        #sub-pixel convolutional layer
        self.conv10 = nn.Conv2d(8, 3 * self.upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
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

        # Third layer
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))

        # Pixel shuffle
        x = self.pixel_shuffle(self.conv10(x))

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


# Create an instance of the model
#model = MultiScale(upscale_factor=2)

# Print the model architecture
#print(model)
