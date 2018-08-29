import random

import torch
import torch.nn as nn
import torch.optim as optim

import constant
from constant import Action
from deep_q_network import DQN


class DinoAgent:
    def __init__(self,device):
        self._device = device 
        self._policy_net = DQN(input_size=constant.IMG_STACK_SIZE,output_size=constant.ACTION_SPACE).to(self._device)
        self._target_net = DQN(input_size=constant.IMG_STACK_SIZE,output_size=constant.ACTION_SPACE).to(self._device)

        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()

        self._criterion = nn.SmoothL1Loss()
        self._optimizer = optim.RMSprop(self._policy_net.parameters(),
                              lr=constant.LEARNING_RATE)
 
 
    def get_action(self,epsilon, state):
        action_t = Action.DO_NOTHING
        if random.random() <= epsilon:  
            action_index = random.randrange(constant.ACTION_SPACE)
        else:
            q = self._policy_net(state.to(self._device))
            action_index = torch.argmax(q).tolist()
        if action_index == 0:
            action_t = Action.DO_NOTHING
        else:
            action_t = Action.JUMP
        return action_t

    def learn(self,samples):
        q_value   = torch.zeros(constant.BATCH, constant.ACTION_SPACE).to(self._device)
        td_target = torch.zeros(constant.BATCH, constant.ACTION_SPACE).to(self._device)

        for i in range(0, constant.BATCH):
            state_t = samples[i][0]
            action_t = samples[i][1]
            reward_t = samples[i][2]
            state_t1 = samples[i][3]
            terminal = samples[i][4]

             

            
            # predict the q value of the next state with the policy network
            predicted_Q_sa_t1 = self._policy_net(state_t1.to(self._device)).detach()
            # get the action which leads to the max q value of the next state
            the_best_action = torch.argmax(predicted_Q_sa_t1).tolist()
            # predict the max q value of the next sate with the target network
            the_optimal_q_value_of_next_state=self._target_net(state_t1.to(self._device))[int(the_best_action)].detach()
            # td(0) 
            td_target[i][int(action_t)] = reward_t if terminal else reward_t + constant.GAMMA * the_optimal_q_value_of_next_state.tolist()

            predicted_Q_sa_t = self._policy_net(state_t.to(self._device))
            q_value[i][int(action_t)] = predicted_Q_sa_t[int(action_t)]

        loss = self._criterion(q_value, td_target)

        self._optimizer.zero_grad()
        loss.backward()

        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()
        
        return loss

    def update_target_net(self):
        self._target_net.load_state_dict(self._policy_net.state_dict())
