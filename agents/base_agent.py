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

import cv2
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
from utils.logger import Logger
from utils.utilis import Utilis


class BaseAgent(object):
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
        self._log_interval = config['GLOBAL'].getint('log_interval')
        self._my_name = self._config['GLOBAL']['working_agent']

    def _get_checkpoint(self):

        best_checkpoint = Utilis.load_best_checkpoint(self._my_name)
        loaded_checkpoint = Utilis.load_checkpoint(self._my_name)

        final_checkpoint = None

        if best_checkpoint is not None:
            final_checkpoint = best_checkpoint
        elif loaded_checkpoint is not None:
            final_checkpoint = loaded_checkpoint

        return final_checkpoint

    def _set_checkpoint(self, t, epoch, epsilon,highest_score, score_t,state_dict):
        
        checkpoint = {
            'time_step': t,
            'epoch': epoch,
            'episilon': epsilon,
            'highest_score': highest_score,
            'state_dict': state_dict
        }

        Utilis.save_checkpoint(checkpoint, highest_score > score_t, self._my_name)

    def _tensorboard_log(self, t, epoch, highest_score,score_t, loss, model):
        info = {'score': score_t, 'hi_score': highest_score,'loss': loss}
        for tag, value in info.items():
            Logger.get_instance().scalar_summary(tag, value, epoch)

        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            Logger.get_instance().histo_summary(tag, value.data.cpu().numpy(), epoch)
            

    def _preprocess_snapshot(self, screenshot):
        transform = transforms.Compose([transforms.CenterCrop((150, 600)),
                                        transforms.Resize((self._img_rows, self._img_columns)),
                                        transforms.Grayscale(),
                                        transforms.ToTensor()])
        return transform(screenshot)

    def _get_game_state(self, game, action):

        screen_shot, reward, terminal, score = game.get_state(action)
        preprocessed_snapshot = self._preprocess_snapshot(screen_shot)
        return preprocessed_snapshot, torch.tensor(reward), torch.tensor(terminal), score

    # A helper mehtod to test the screenshots  during training
    def _show(self, img, img2):
        npimg1 = img.numpy()
        npimg2 = img2.numpy()

        plt.subplot(211)
        plt.imshow(np.transpose(npimg1, (1, 2, 0)))
        plt.subplot(212)
        plt.imshow(np.transpose(npimg2, (1, 2, 0)))
        plt.show()
