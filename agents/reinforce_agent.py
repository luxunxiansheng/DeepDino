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

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from agents.base_agent import BaseAgent
from model.deep_mind_network_base import DeepMindNetworkBase
from utils.utilis import Utilis


class REINFORCEAgent(BaseAgent):
    def __init__(self, config):
        super(REINFORCEAgent, self).__init__(config)

        self._lr = config['REINFOCE'].getfloat('learning_rate')

        self._policy_network_name = config['REINFOCE'].get('policy_model_name')
        self._policy_net = DeepMindNetworkBase.create(self._policy_network_name, input_channels=self._image_stack_size, output_size=self._action_space).cuda()
        self._policy_optimizer = optim.Adam(self._policy_net.parameters(), lr=self._lr)

        self._state_value_network_name = config['REINFOCE'].get('state_value_model_name')
        self._state_value_net = DeepMindNetworkBase.create(self._state_value_network_name, input_channels=self._image_stack_size, output_size=1).cuda()
        self._state_value_optimizer = optim.Adam(self._state_value_net.parameters(), lr=self._lr)
        

    def _predict_state_value(self, state):
        state_value = self._state_value_net(state.cuda())
        return state_value

    def _run_episode(self, game, t, init_state):

        episode_log_prob_actions = []
        episode_rewards = []
        episode_state_value=[]

        current_state = init_state
        # run one episode until done
        while (True):

            state_value = self._predict_state_value(current_state)
            episode_state_value.append(state_value)

            log_prob_action, action_t = self._get_action(current_state)
            episode_log_prob_actions.append(log_prob_action)

            # run the selected action and observe next screenshot & reward
            next_screentshot, reward_t, terminal, score_t = self._game_step_forward(game, action_t)
            episode_rewards.append(reward_t)

            if terminal:
                break
            else:
                # assemble the next state  which contains the lastest 3 screenshot  and the next screenshot
                current_state = self._get_next_state(current_state, next_screentshot)
                t = t+1

        return episode_log_prob_actions, episode_rewards, episode_state_value,score_t, t

    def train(self, game):
        t = 0
        epoch = 0
        highest_score = 0
        state_dict = None

        # resume from the checkpoint
        checkpoint = self._get_checkpoint()
        if self._config['GLOBAL'].getboolean('resume') and checkpoint is not None:
            t = checkpoint['time_step']
            epoch = checkpoint['epoch']
            highest_score = checkpoint['highest_score']
            state_dict = checkpoint['state_dict']
            self._policy_net.load_state_dict(state_dict)

        screenshot, _, _, _ = self._game_step_forward(game, 0)
        # the first state containes the first 4 frames
        initial_state = torch.stack((screenshot, screenshot, screenshot, screenshot))

        # run episodes again and again
        while (True):
            
            episode_log_prob_actions, episode_rewards, episode_state_value,final_score, t = self._run_episode(game, t, initial_state)

            is_best = False
            if final_score > highest_score:
                highest_score = final_score
                is_best = True

            loss = self._learn(episode_log_prob_actions, episode_rewards,episode_state_value,epoch)

            checkpoint = {
                'time_step': t,
                'epoch': epoch,
                'highest_score': highest_score,
                'state_dict': self._policy_net.state_dict()
            }

            print("t:", t, "epoch:", epoch, "loss:", loss)
            Utilis.save_checkpoint(checkpoint, is_best, self._my_name)

            self._tensorboard_log(t, epoch, highest_score, final_score, loss, self._policy_net)

            epoch = epoch + 1

            

    def _learn(self, episode_log_prob_actions, episode_rewards,epsode_state_value,epoch):
        
        eps= np.finfo(np.float32).eps.item()
        # the return at time t
        G_t = 0
        G_t_list = []
        for r in episode_rewards[::-1]:
            G_t = r + self._gamma * G_t
            G_t_list.insert(0, G_t)
        G_t_tensor = torch.tensor(G_t_list).cuda()

        G_t_tensor = (G_t_tensor-G_t_tensor.mean())/(G_t_tensor.std()+eps)

        policy_loss = []
        state_value_loss = []
        
        for state_value, log_prob, g in zip(epsode_state_value,episode_log_prob_actions,G_t_tensor):
            # subtract the state value from the G. 
            policy_loss.append(-log_prob * (g - state_value.detach()))
            
            # a biased estimation of state value with the q value of a single trajectory 
            state_value_loss.append(F.l2(state_value,g))

        
        self._state_value_optimizer.zero_grad()
        state_value_loss = torch.stack(state_value_loss, dim=0).mean()
        state_value_loss.backward()
        for param in self._state_value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._state_value_optimizer.step()

        
        if epoch % 1 == 0:
            self._policy_optimizer.zero_grad()
            policy_loss = torch.stack(policy_loss, dim=0).sum()
            policy_loss.backward()
            for param in self._policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self._policy_optimizer.step()

        return state_value_loss.tolist()

    def _get_action(self, state):
        probs, log_probs = self._policy_net(state.cuda())
        m = Categorical(probs)
        action_index = m.sample()
        return log_probs[action_index.item()], action_index.item()
