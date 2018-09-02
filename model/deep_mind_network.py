import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F


class DeepMindNetwork(nn.Module):
    '''
    The convolution newtork proposed by Mnih at al(2015) 
    in the paper "Playing Atari with Deep Reinforcement 
    Learning"
    
    '''

    def __init__(self,input_channels,output_size):
        
        super(DeepMindNetwork, self).__init__()
        
        self._input_channels = input_channels
        self._output_size = output_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(self._input_channels, 32, kernel_size=8,stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.fc1=nn.Sequential(
            nn.Linear(7*7*64,256),
            nn.ReLU()
        )

        self.fc2=nn.Sequential(
            nn.Linear(256,self._output_size),
            
        )
    
           
    def forward(self, input):
        x = torch.transpose(input,0,1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(1,-1)

        x = self.fc1(x)
        x = self.fc2(x) 
        
        return torch.squeeze(x)



    


