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
 #/  

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from agents.base_agent import BaseAgent
from model.deep_mind_network_base import DeepMindNetworkBase
from utils.utilis import Utilis


class ActorCriticAgent(BaseAgent):
    def __init__(self, config):
        super(ActorCriticAgent, self).__init__(config)
        
        self._lr= config['ACTOR-CRITIC'].getfloat('learning_rate')
        self._network_name = config['ACTOR-CRITIC'].get('model_name')

        self._policy_net = DeepMindNetworkBase.create(self._network_name, input_channels=self._image_stack_size, output_size=self._action_space).cuda()
        self._optimizer = optim.Adam(self._policy_net.parameters(), lr=self._lr)

        
