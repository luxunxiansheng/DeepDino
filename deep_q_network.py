import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F

import constant

class Deep_Q__network(nn.Module):
    def __init__(self):
        
        super(Deep_Q__network, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, 8),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4),
            nn.MaxPool2d(2),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 2),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.fc1=nn.Sequential(
            nn.Linear(3136,512),
            nn.ReLU()
        )

        self.fc2=nn.Sequential(
            nn.Linear(512,2),
            nn.ReLU()
        )

          

        self.initialize_weights()
    
        


    def forward(self, input):
        x = torch.transpose(input,0,1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(1,-1)

        x = self.fc1(x)
        x = self.fc2(x) 
        
        return torch.squeeze(x)



    def initialize_weights(self):
        for m in self._modules:
            if isinstance(m,nn.Conv2d):
                m.weight.data.normal_(0,0.02)
                m.bias.data.zero_()


