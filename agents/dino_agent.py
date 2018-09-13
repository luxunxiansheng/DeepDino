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


import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torchvision import transforms

from common.action import Action
from common.replay_memory import Replay_Memory
from model.deep_mind_network import DeepMindNetwork
from utils.utilis import Utilis


class DinoAgent(object):
    @staticmethod
    def create(config):
        if config['GLOBAL']['working_agent'] == 'DQNAgent':
            from agents.dqn_agent import DQNAgent   # dynamic import. Refer to the Item 52: know how to break circular dependencey in book  "Effective Python"
            return DQNAgent(config)


    def __init__(self, config):
        self._config = config
        self._image_stack_size = config['GLOBAL'].getint('img_stack_size')
        self._action_space = config['GLOBAL'].getint('action_space')
        self._img_rows = config['GLOBAL'].getint('img_rows')
        self._img_columns = config['GLOBAL'].getint('img_columns')
        self._log_interval= config['GLOBAL'].getint('log_interval')
