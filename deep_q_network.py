import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F

import constant


class DQN(nn.Module):
    def __init__(self,input_size,output_size):
        
        super(DQN, self).__init__()
        
        self._input_size  = input_size
        self._output_size = output_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(self._input_size, 32, 8,stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.fc1=nn.Sequential(
            nn.Linear(7*7*64,512),
            nn.ReLU()
        )

        self.fc2=nn.Sequential(
            nn.Linear(512,self._output_size),
            
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



    


