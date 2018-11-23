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
from common.replay_memory import Replay_Memory
from model.deep_mind_network_base import DeepMindNetworkBase
from utils.utilis import Utilis


class DQNAgent(BaseAgent):
    def __init__(self, config):

        super(DQNAgent, self).__init__(config)

        self._batch_size = config['DQN'].getint('batch')
        self._isDQN = config['DQN'].getboolean('dqn')

        self._momentum = config['DQN'].getfloat('momentum')
        self._lr = config['DQN'].getfloat('learning_rate')

        self._explore = config['DQN'].getint('explore')
        self._replay_memory_capacity = config['DQN'].getint('replay_memory_capacity')
        self._update_target_interval = config['DQN'].getint('update_target_interval')

        self._observations = config['DQN'].getint('observations')

        self._network_name = config['DQN'].get('model_name')

        self._policy_net = DeepMindNetworkBase.create(self._network_name, input_channels=self._image_stack_size, output_size=self._action_space).cuda()
        self._target_net = DeepMindNetworkBase.create(self._network_name, input_channels=self._image_stack_size, output_size=self._action_space).cuda()

        self._criterion = nn.SmoothL1Loss()
        self._optimizer = optim.RMSprop(self._policy_net.parameters(), momentum=self._momentum, lr=self._lr)


    def _get_exploration_method(self):
        if self._network_name == "NoisyNetwork":
            return "ParameterNoisy"
        else:
            return "ActionNoisy"
            

    def _explore_with_noisy_network(self, state):
        q = self._policy_net(state.cuda())
        return  torch.argmax(q).tolist()    

    def _explore_with_e_greedy(self, epsilon, state):
        if random.random() <= epsilon:
            return  random.randrange(self._action_space)
        else:
            q = self._policy_net(state.cuda())
            return  torch.argmax(q).tolist()    

    # greedy policy
    def _get_optimal_action(self, state):
        
        q = self._policy_net(state.cuda())
        return  torch.argmax(q).tolist()
    
    

    def _predict_optimal_Q_value_with_DoubleDQN(self, state_t1):
        # predict the q value of the next state with the policy network
        predicted_Q_sa_t1 = self._policy_net(state_t1.cuda()).detach()
        # get the action which leads to the max q value of the next state
        the_best_action = torch.argmax(predicted_Q_sa_t1)
        # predict the max q value of the next sate with the target network
        the_optimal_q_value_of_next_state = self._target_net(state_t1.cuda())[int(the_best_action)].detach()

        return the_optimal_q_value_of_next_state

    def _predict_optimal_Q_value_with_DQN(self, state_t1):
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

            the_optimal_q_value_of_next_state = self._predict_optimal_Q_value_with_DQN(stack_t1) if self._isDQN else self._predict_optimal_Q_value_with_DoubleDQN(stack_t1)

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

    def train(self, game):
        epsilon = self._init_epsilon
        t = 0
        epoch = 0
        highest_score = 0
        state_dict = None

        # resume from the checkpoint
        checkpoint = self._get_checkpoint()
        if self._config['GLOBAL'].getboolean('resume') and checkpoint is not None:
            t = checkpoint['time_step']
            epoch = checkpoint['epoch']
            epsilon = checkpoint['epsilon']
            highest_score = checkpoint['highest_score']

            state_dict = checkpoint['state_dict']
            self._policy_net.load_state_dict(state_dict)

        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()

        replay_memory = Replay_Memory(self._replay_memory_capacity)

        screenshot, _, _, _ = self._game_step_forward(game, 0)

        # the first state containes the first 4 frames
        initial_state = torch.stack((screenshot, screenshot, screenshot, screenshot))

        current_state = initial_state

        while (True):  # endless running
            loss = 0
            reward_t = 0
            action_t =0

            
            if "ActionNoisy" == self._get_exploration_method():
                # choose an action epsilon greedy
                action_t = self._explore_with_e_greedy(epsilon, current_state)

                # reduced the epsilon (exploration parameter) gradually
                if epsilon > self._final_epsilon:
                    epsilon -= (self._init_epsilon-self._final_epsilon) / self._explore
            
            else:
                action_t=self._explore_with_noisy_network(current_state)
            

            # run the selected action and observe next screenshot & reward
            next_screentshot, reward_t, terminal, score_t = self._game_step_forward(game, action_t)

            # assemble the next state  which contains the lastest 3 screenshot  and the next screenshot
            next_state = self._get_next_state(current_state, next_screentshot)

            """ 
            grid_img = torchvision.utils.make_grid(current_state, 4, padding=10)
            grid_img2 = torchvision.utils.make_grid(next_state, 4, padding=10)
            self._show(grid_img,grid_img2) 
            
            """

            # store the transition in experience_replay_memory
            replay_memory.push((current_state, action_t, reward_t, next_state, terminal))

            # Not to learn and log until the replay memory is not too empty
            if replay_memory.size() > self._observations:
                experience_batch = replay_memory.sample(self._batch_size)
                loss = self._learn(experience_batch)

                if t % self._log_interval == 0:
                    print("t:", t, "epoch:", epoch, "loss:", loss)

                if t % self._update_target_interval == 0:
                    self._update_target_net()

            if terminal:
                current_state = initial_state
                is_best=False

                if score_t > highest_score:
                    highest_score = score_t
                    is_best=True
                          
                checkpoint = {
                    'time_step': t,
                    'epoch': epoch,
                    'epsilon': epsilon,
                    'highest_score': highest_score,
                    'state_dict': self._policy_net.state_dict()
                }
                Utilis.save_checkpoint(checkpoint, is_best,self._my_name)
                self._tensorboard_log(t, epoch, highest_score, score_t, loss, self._policy_net)
               
                epoch = epoch + 1
            else:
                current_state = next_state

            t = t + 1

    def replay(self, game):

        t = 0

        state_dict = None

        # resume from the checkpoint
        checkpoint = self._get_checkpoint()
        if checkpoint is None:
            return

        state_dict = checkpoint['state_dict']
        self._policy_net.load_state_dict(state_dict)

        # init the start state
        screenshot, _, _, _ = self._game_step_forward(game, 0)

        # the first state stack containes the first 4 frames
        initial_state = torch.stack((screenshot, screenshot, screenshot, screenshot))

        current_state = initial_state

        while (True):  # endless running
            action_t = 0

            
            if t % self._frame_per_action == 0:  # parameter to skip frames for actions
                action_t = self._get_optimal_action(current_state)

            # run the selected action and observed next state and reward
            next_screenshot, _, terminal, _ = self._game_step_forward(game, action_t)

            # assemble the next state stack which contains the lastest 3 states and the next state
            next_state = self._get_next_state(current_state, next_screenshot)

            if terminal:
                current_state = initial_state

            else:
                current_state = next_state

            t = t + 1
