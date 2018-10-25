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

from operator import add, sub

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from agents.base_agent import BaseAgent
from model.deep_mind_network_base import DeepMindNetworkBase
from utils.utilis import Utilis


class ActorCriticAgent(BaseAgent):
    def __init__(self, config):
        
        super(ActorCriticAgent, self).__init__(config)
        
        self._lr = config['ACTOR-CRITIC'].getfloat('learning_rate')

        self._policy_network_name = config['ACTOR-CRITIC'].get('policy_model_name')
        self._policy_net = DeepMindNetworkBase.create(self._policy_network_name, input_channels=self._image_stack_size, output_size=self._action_space).cuda()
        self._policy_optimizer = optim.Adam(self._policy_net.parameters(), lr=self._lr)

        self._state_value_network_name = config['ACTOR-CRITIC'].get('state_value_model_name')
        self._state_value_net = DeepMindNetworkBase.create(self._state_value_network_name, input_channels=self._image_stack_size, output_size=1).cuda()
                
        self._state_value_optimizer = optim.Adam(self._state_value_net.parameters(), lr=self._lr)
    

    def _get_action(self, state):
        probs, log_probs = self._policy_net(state.cuda())
        distribution = Categorical(probs)
        action_index = distribution.sample()
        return log_probs[action_index.item()], action_index.item(),distribution.entropy()
    
    
    def _predict_state_value_with_netrual_network(self,state):
        return self._state_value_net(state.cuda())

    def _run_steps(self, game, t,current_state,n_step):
        log_prob_actions = []
        rewards = []
        states = []
        entropys=[]
                       
        for _ in range(n_step):

            states.append(current_state) 
         
            log_prob_action, action_t, entropy = self._get_action(current_state)
            log_prob_actions.append(log_prob_action)
            entropys.append(entropy)

            # run the selected action and observe next screenshot & reward
            next_screentshot, reward_t, terminal, score_t = self._game_step_forward(game, action_t)
            rewards.append(reward_t.cuda())

            if terminal:
                current_state = None
                break
            else:
                # assemble the next state  which contains the lastest 3 screenshot  and the next screenshot
                current_state = self._get_next_state(current_state, next_screentshot)
                t = t+1
        return current_state,score_t,log_prob_actions,entropys,rewards,states,t
        
         
    def _fit_state_value_model(self, state_value_losses):
        self._state_value_optimizer.zero_grad()
        state_value_loss = torch.stack(state_value_losses, dim=0).mean()
        state_value_loss.backward()
        for param in self._state_value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._state_value_optimizer.step()

        return state_value_loss.item()

    def _improve_policy(self,policy_losses):
        self._policy_optimizer.zero_grad()
        policy_loss = torch.stack(policy_losses, dim=0).sum()
        policy_loss.backward()
        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._policy_optimizer.step()   
         
    
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
            # Run One Episode
            is_best=False
            fisrt_step_state = initial_state
            episode_final_score = 0
            episode_value_loss=0
            while(True):
                # run steps following the pi alone the trajectory    
                last_step_state,last_step_score,log_prob_actions,entropys,rewards,states,t = self._run_steps(game, t, fisrt_step_state, 5)
               
                returns= 0 if last_step_state is None else self._predict_state_value_with_netrual_network(last_step_state)

                policy_losses = []
                state_value_losses = []
                                                
                for log_prob_action,reward,entropy,state in zip(log_prob_actions[::-1], rewards[::-1],entropys[::-1],states[::-1]):
                    returns = reward+ self._gamma * returns 
                    state_value= self._predict_state_value_with_netrual_network(state)
                    state_value_losses.append(F.smooth_l1_loss(state_value,returns.detach()))

                    advantage= returns-state_value
                    policy_losses.append(-log_prob_action * advantage.detach()+entropy)
                 
                
                episode_value_loss=self._fit_state_value_model(state_value_losses)
                self._improve_policy(policy_losses)

                if last_step_state is None:
                    episode_final_score=last_step_score
                    if last_step_score > highest_score:
                        highest_score = last_step_score
                        is_best = True
                
                    break
                else:
                    fisrt_step_state = last_step_state

            checkpoint = {
                'time_step': t,
                'epoch': epoch,
                'highest_score': highest_score,
                'state_dict': self._policy_net.state_dict()
            }

            print("t:", t, "epoch:", epoch, "loss:", episode_value_loss)
            Utilis.save_checkpoint(checkpoint, is_best, self._my_name)

            self._tensorboard_log(t, epoch, highest_score, episode_final_score, episode_value_loss, self._policy_net)

            epoch = epoch + 1
