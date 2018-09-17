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

from agents.base_agent import BaseAgent
from common.action import Action
from common.replay_memory import Replay_Memory
from model.base_network import BaseNetwork
from utils.utilis import Utilis


class DQNAgent(BaseAgent):
    def __init__(self, config):

        super(DQNAgent, self).__init__(config)

        self._batch_size = config['DQN'].getint('batch')
        self._isDQN = config['DQN'].getboolean('dqn')
        self._gamma = config['DQN'].getfloat('gamma')
        self._momentum = config['DQN'].getfloat('momentum')
        self._lr = config['DQN'].getfloat('learning_rate')
        self._final_epsilon = config['DQN'].getfloat('final_epsilon')
        self._init_epsilon = config['DQN'].getfloat('init_epsilon')
        self._explore = config['DQN'].getint('explore')
        self._replay_memory_capacity = config['DQN'].getint('replay_memory_capacity')
        self._update_target_interval = config['DQN'].getint('update_target_interval')
        self._frame_per_action = config['DQN'].getint('frame_per_action')
        self._observations = config['DQN'].getint('observations')

        self._network_name = config['DQN'].get('model_name')

        self._policy_net = BaseNetwork.create(self._network_name, input_channels=self._image_stack_size, output_size=self._action_space).cuda()
        self._target_net = BaseNetwork.create(self._network_name, input_channels=self._image_stack_size, output_size=self._action_space).cuda()

        self._criterion = nn.SmoothL1Loss()
        self._optimizer = optim.RMSprop(self._policy_net.parameters(), momentum=self._momentum, lr=self._lr)

    
    # e-greedy exploration 
    def _get_action(self, epsilon, state):
        action_t = Action.DO_NOTHING
        if random.random() <= epsilon:
            action_index = random.randrange(self._action_space)
        else:
            q = self._policy_net(state.cuda())
            action_index = torch.argmax(q).tolist()
        if action_index == 0:
            action_t = Action.DO_NOTHING
        else:
            action_t = Action.JUMP
        return action_t

    def _Double_DQN(self, state_t1):
        # predict the q value of the next state with the policy network
        predicted_Q_sa_t1 = self._policy_net(state_t1.cuda()).detach()
        # get the action which leads to the max q value of the next state
        the_best_action = torch.argmax(predicted_Q_sa_t1)
        # predict the max q value of the next sate with the target network
        the_optimal_q_value_of_next_state = self._target_net(state_t1.cuda())[int(the_best_action)].detach()

        return the_optimal_q_value_of_next_state

    def _DQN(self, state_t1):
        predicted_Q_sa_t1 = self._target_net(state_t1.cuda()).detach()
        the_optimal_q_value_of_next_state = torch.max(predicted_Q_sa_t1)
        return the_optimal_q_value_of_next_state

    def _learn(self, samples):

        q_value = torch.zeros(self._batch_size, self._action_space).cuda()
        td_target = torch.zeros(self._batch_size, self._action_space).cuda()

        for i in range(0, self._batch_size):
            stack_t = samples[i][0]
            action_t = samples[i][1]
            reward_t = samples[i][2]
            stack_t1 = samples[i][3]
            terminal = samples[i][4]

            the_optimal_q_value_of_next_state = self._DQN(stack_t1) if self._isDQN else self._Double_DQN(stack_t1)

            # td(0)
            td_target[i][int(action_t)] = reward_t if terminal else reward_t + self._gamma*the_optimal_q_value_of_next_state.tolist()

            
            predicted_Q_sa_t = self._policy_net(stack_t.cuda())
            q_value[i][int(action_t)] = predicted_Q_sa_t[int(action_t)]

        loss = self._criterion(q_value, td_target)

        self._optimizer.zero_grad()
        loss.backward()

        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()

        return loss.tolist()

    def _update_target_net(self):
        self._target_net.load_state_dict(self._policy_net.state_dict())

    def _get_most_recent_states(self, the_privous_most_recnent_frames, next_frame):
        the_most_most_recent_frames = the_privous_most_recnent_frames.clone()
        the_most_most_recent_frames[0:-1] = the_privous_most_recnent_frames[1:]
        the_most_most_recent_frames[-1] = next_frame
        return the_most_most_recent_frames

    def train(self, game):

        epsilon = self._init_epsilon
        t = 0
        epoch = 0
        highest_score = 0
        state_dict = None

        # resume from the checkpoint
        checkpoint = self._get_checkpoint()
        if checkpoint is not None:
            t = checkpoint['time_step']
            epoch = checkpoint['epoch']
            epsilon = checkpoint['epsilon']
            highest_score= checkpoint['highest_score']
            
            state_dict = checkpoint['state_dict']
            self._policy_net.load_state_dict(state_dict)

        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()

        replay_memory = Replay_Memory(self._replay_memory_capacity)

        # init the start state
        state_t, _, _, _ = self._get_game_state(game, Action.DO_NOTHING)

        # the first state stack containes the first 4 frames
        initial_state_stack = torch.stack((state_t, state_t, state_t, state_t))

        the_most_recent_state_stack = initial_state_stack

        while (True):  # endless running
            loss = 0

            reward_t = 0
            action_t = Action.DO_NOTHING

            # choose an action epsilon greedy
            if t % self._frame_per_action == 0:  # parameter to skip frames for actions
                action_t = self._get_action(epsilon, the_most_recent_state_stack)

            # reduced the epsilon (exploration parameter) gradually
            if epsilon > self._final_epsilon:
                epsilon -= (self._init_epsilon-self._final_epsilon) / self._explore

            # run the selected action and observed next state and reward
            state_t1, reward_t, terminal, score_t = self._get_game_state(game, action_t)
           

            # assemble the next state stack which contains the lastest 3 states and the next state
            the_most_most_recent_state_stack = self._get_most_recent_states(the_most_recent_state_stack, state_t1)

            """ 
            grid_img = torchvision.utils.make_grid(the_most_recent_state_stack, 4, padding=10)
            grid_img2 = torchvision.utils.make_grid(the_most_most_recent_state_stack, 4, padding=10)
            self._show(grid_img,grid_img2) 
            
            """

            # store the transition in experience_replay_memory
            replay_memory.push((the_most_recent_state_stack, action_t, reward_t, the_most_most_recent_state_stack, terminal))

            # Not to learn and log until the replay memory is not too empty
            if replay_memory.size() > self._observations:
                experience_batch = replay_memory.sample(self._batch_size)
                loss = self._learn(experience_batch)

                if t%  self._log_interval == 0:
                    print("t:", t, "epoch:", epoch, "loss:", loss)

                if t % self._update_target_interval == 0:
                    self._update_target_net()

            if terminal:
                the_most_recent_state_stack = initial_state_stack

                if score_t > highest_score:
                    highest_score = score_t

                game.pause()
                # log and save the checkpoint
                self._set_checkpoint(t, epoch, epsilon, highest_score, score_t, self._policy_net.state_dict())
                self._tensorboard_log(t, epoch, highest_score, score_t, loss, self._policy_net)
                game.resume()

                epoch = epoch + 1
            else:
                the_most_recent_state_stack = the_most_most_recent_state_stack

            t = t + 1
