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
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2) #padding = (kernelsize-1)/2 to ensure same resolution as input
        self.conv2 = nn.Conv2d(16, 16, kernel_size=9, padding=4)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=13, padding=6)
        

        # second layer
        self.conv4 = nn.Conv2d(48, 8, kernel_size=5, padding=2)
        #self.conv4 = nn.Conv2d(48, 24, kernel_size=5, padding=2)  # Assuming x2_1, x2_2, and x2_3 each have 8 channels
        self.conv5 = nn.Conv2d(8, 8, kernel_size=9, padding=4)
        self.conv6 = nn.Conv2d(8, 8, kernel_size=13, padding=6)


        #sub-pixel convolutional layer
        self.conv7 = nn.Conv2d(24, 3 * self.upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)

        


        self._initialize_weights()

    def forward(self, x):

        #first layer
        x1_1 = self.relu(self.conv1(x))
        x1_2 = self.relu(self.conv2(x1_1))
        x1_3 = self.relu(self.conv3(x1_2))
        x1 = torch.cat((x1_1, x1_2, x1_3), dim=1)

        #second layer
        x2_1 = self.relu(self.conv4(x1))
        x2_2 = self.relu(self.conv5(x2_1))
        x2_3 = self.relu(self.conv6(x2_2))
        x2 = torch.cat((x2_1, x2_2, x2_3), dim=1)

        #pixel shuffle
        x3 = self.pixel_shuffle(self.conv7(x2))
    

        return x3

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
# Create an instance of the model
model = Net(upscale_factor=2)

# Print the model architecture
#print(model)
