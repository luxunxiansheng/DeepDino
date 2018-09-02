# #### BEGIN LICENSE BLOCK #####
# Version: MPL 1.1/GPL 2.0/LGPL 2.1
#
# The contents of this file are subject to the Mozilla Public License Version
# 1.1 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/
#
# Software distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
# for the specific language governing rights and limitations under the
# License.
#
#
# Contributor(s):
#
#    Bin.Li (ornot2008@yahoo.com)
#
#
# Alternatively, the contents of this file may be used under the terms of
# either the GNU General Public License Version 2 or later (the "GPL"), or
# the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
# in which case the provisions of the GPL or the LGPL are applicable instead
# of those above. If you wish to allow use of your version of this file only
# under the terms of either the GPL or the LGPL, and not to allow others to
# use your version of this file under the terms of the MPL, indicate your
# decision by deleting the provisions above and replace them with the notice
# and other provisions required by the GPL or the LGPL. If you do not delete
# the provisions above, a recipient may use your version of this file under
# the terms of any one of the MPL, the GPL or the LGPL.
#
# #### END LICENSE BLOCK #####
#
# /


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

    def __init__(self, input_channels, output_size):

        super(DeepMindNetwork, self).__init__()

        self._input_channels = input_channels
        self._output_size = output_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(self._input_channels, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(7*7*64, 256),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, self._output_size),

        )

    def forward(self, input):
        x = torch.transpose(input, 0, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(1, -1)

        x = self.fc1(x)
        x = self.fc2(x)

        return torch.squeeze(x)
