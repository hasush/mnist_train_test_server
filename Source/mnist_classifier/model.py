import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        
        # Convolutional layers.
        self.conv1=nn.Conv2d(in_channels=1,
                             out_channels=32*2,
                             kernel_size=3,
                             stride=1)
        self.conv2=nn.Conv2d(in_channels=32*2,
                             out_channels=128*2*2,
                             kernel_size=3,
                             stride=1)
        self.conv3=nn.Conv2d(in_channels=128*2*2,
                             out_channels=128*2*2*2,
                             kernel_size=3,
                             stride=1)
        self.pool=nn.MaxPool2d(2,2)

        # Linear layers.
        self.fc1=nn.Linear(in_features=128*2*2*2,out_features=512)
        self.fc2=nn.Linear(in_features=512,out_features=128)
        self.output=nn.Linear(in_features=128,out_features=10)

    def forward(self, x):

        # Convolutional layers.
        x=self.conv1(x)
        x=F.relu(x)
        x=self.pool(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=self.pool(x)
        x=self.conv3(x)
        x=F.relu(x)
        x=self.pool(x)

        # Linear layers with log softmax activation.
        x=torch.flatten(x,1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.output(x)
        x=F.log_softmax(x, dim=1)

        return x