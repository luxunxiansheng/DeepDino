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
        x = self.conv1(input)
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


"""     def learn(self,samples):
                

        q_value= torch.zeros(constant.BATCH)
        td_target=torch.zeros([constant.BATCH])
        
        for i in range(0,constant.BATCH):
            state_t  = samples[i][0]
            action_t = samples[i][1]
            reward_t = samples[i][2]
            state_t1 = samples[i][3]
            terminal = samples[i][4]

                        
            # td(0) 
            Q_sa_t1 = self.forward(state_t1) 
            td_target[i] = reward_t if terminal else reward_t+ constant.GAMMA * torch.max(Q_sa_t1).tolist()
         

            Q_sa_t = self.forward(state_t)
            q_value[i]   = Q_sa_t[int(action_t)]
  
            
        loss= self.criterion(q_value,td_target).cuda()
                
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss 
 """