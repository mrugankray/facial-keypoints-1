## TODO: define the convolutional neural network architecture

import torch
import torch.cuda
#from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I



def gpu(x):
    return x.cuda() if torch.cuda.is_available() else x


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        self.conv1 = nn.Conv2d(1, 32, 5)        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        self.pool = nn.MaxPool2d(2, 2)          ## output size = (W-F)/S +1 = (220-2)/2 +1 = 110
        
        self.conv2 = nn.Conv2d(32, 64, 3)       ## output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        # pool
        self.conv3 = nn.Conv2d(64, 128, 3, 2)   ## output size = (W-F)/S +1 = (54-3)/2 +1 = 26
        # pool
        self.conv4 = nn.Conv2d(128, 256, 3, 2)  ## output size = (W-F)/S +1 = (13-3)/1 +1 = 6
        # pool
        self.conv5 = nn.Conv2d(256, 512, 3)     ## output size = (W-F)/S +1 = (3-3)/1 +1 = 1
        
        self.fc1 = nn.Linear(512, 512)
        self.fc1_drop = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(512, 136)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # Apply convolutional layers
        x = self.pool(F.selu(self.conv1(x)))
        x = self.pool(F.selu(self.conv2(x)))
        x = self.pool(F.selu(self.conv3(x)))
        x = self.pool(F.selu(self.conv4(x)))
        x = F.selu(self.conv5(x))

        # Flatten and continue with dense layers
        x = x.view(x.size(0), -1)
        x = F.selu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
