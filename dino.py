
import io
import json
import os
import pickle
import random
import time
from collections import deque

from random import randint

import pandas as pd
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import constant
from constant import Action
from game import Game
from deep_q_network import Deep_Q__network,DQN

from torchvision.utils import make_grid
import matplotlib.pyplot as plt 
import numpy as np 

  


def show(img):
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
    plt.close()

def learn(samples,q_value_network,optimizer,criterion):
    
    q_value= torch.zeros(constant.BATCH,2).cuda()
    td_target=torch.zeros(constant.BATCH,2).cuda()
        
    for i in range(0,constant.BATCH):
        state_t  = samples[i][0]
        action_t = samples[i][1]
        reward_t = samples[i][2]
        state_t1 = samples[i][3]
        terminal = samples[i][4]
                       
        # td(0) 
        predicted_Q_sa_t1 = q_value_network(state_t1.cuda())
        td_target[i][int(action_t)] = reward_t if terminal else reward_t+ constant.GAMMA * torch.max(predicted_Q_sa_t1).tolist()
   
        predicted_Q_sa_t = q_value_network(state_t.cuda())
        q_value[i][int(action_t)]  = predicted_Q_sa_t[int(action_t)]
  
       
    loss= criterion(q_value,td_target).cuda()
                
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss 




def train(game, deep_q_network):
   
    t = 0

    experience_replay_memory = deque()
    epsilon = constant.INITIAL_EPSILON
      
    # get next step after performing the action
    image_t, _, terminal = game.get_state(Action.DO_NOTHING)
    initial_state =  torch.stack((image_t, image_t, image_t, image_t))
    
    state_t = initial_state.clone()
    state_t1 = initial_state.clone()    
       
    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(deep_q_network.parameters(), lr=constant.LEARNING_RATE)
    
    while (True):  # endless running
        loss = 0
        reward_t = 0
        action_t = Action.DO_NOTHING
        # choose an action epsilon gredy
        if t % constant.FRAME_PER_ACTION == 0:  # parameter to skip frames for actions
            action_t = e_greedy(epsilon, deep_q_network, state_t)

        # We reduced the epsilon (exploration parameter) gradually
        if epsilon > constant.FINAL_EPSILON:
            epsilon -= (constant.INITIAL_EPSILON -
                        constant.FINAL_EPSILON) / constant.EXPLORE

        # run the selected action and observed next state and reward
        image_t1, reward_t, terminal = game.get_state(action_t)
        
        state_t1[0] = state_t[1].clone()
        state_t1[1] = state_t[2].clone()
        state_t1[2] = state_t[3].clone()
        state_t1[3] =   image_t1.clone()
        
        #show(make_grid([state_t[0],state_t[1],state_t[2],state_t[3],state_t1[0],state_t1[1],state_t1[2],image_t1],nrow=4,padding=10))
 
           
        # store the transition in experience_replay_memory
        experience_replay_memory.append((state_t, action_t, reward_t, state_t1, terminal))

        memory_len = len(experience_replay_memory)

        if memory_len > constant.REPLAY_MEMORY:
            experience_replay_memory.popleft()

        if(t>constant.OBSERVATION):
            state_batch = random.sample(experience_replay_memory, constant.BATCH)
            loss=learn(state_batch,deep_q_network,optimizer,criterion)
            
            if t%10==0:
                print("t:",t,  "loss:", loss.tolist())
      
        
        if terminal:
            state_t = initial_state.clone() 
        else: 
            state_t = state_t1.clone()
        
        t = t + 1
   

def e_greedy(epsilon, q_value_network, state):

    action_t = Action.DO_NOTHING

    if random.random() <= epsilon:  # randomly explore an action
       # print("----------Random Action----------")
        action_index = random.randrange(len(Action))

    else:  
       # print("----------Greedy Action----------")
        q = q_value_network(state.cuda())
        action_index = torch.argmax(q).tolist()
    
    if action_index == 0:
        action_t = Action.DO_NOTHING
    else:
        action_t = Action.JUMP   
    return action_t


def main():

    gpus = [0, 1, 2, 3]
    torch.cuda.set_device(gpus[3])

    model = DQN()
    model.cuda()
  
    game = Game()

    try:
        train(game, model)
    except StopIteration:
        game.end()


if __name__ == '__main__':
    main()
