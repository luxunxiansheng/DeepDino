import io
import json
import os
import pickle
import random
import time
from collections import deque
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torchvision.utils import make_grid

import constant
from constant import Action
from deep_q_network import DQN
from game import Game
from dino import DinoAgent

loss_file_path = "/home/lb/workspace/Dino/objects/loss_df.csv"
actions_file_path = "/home/lb/workspace/Dino/objects/actions_df.csv"
q_value_file_path = "/home/lb/workspace/Dino/objects/q_values.csv"
score_file_path = "/home/lb/workspace/Dino/objects/scores_df.csv"
model_file_path = "/home/lb/workspace/Dino/model/dqn.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.close()


def _get_game_state(game, action):
    transform = transforms.Compose([transforms.CenterCrop((150, 600)), transforms.Resize(
        (constant.IMG_ROWS, constant.IMG_COLS)), transforms.Grayscale(), transforms.ToTensor()])
    screen_shot, reward, terminal = game.get_state(action)
    return transform(screen_shot), torch.tensor(reward), torch.tensor(terminal)


def train(game,agent):
    t = 0
    experience_replay_memory = deque()
    epsilon = constant.INITIAL_EPSILON

    # get next step after performing the action
    image_t, _, terminal = _get_game_state(game, Action.DO_NOTHING)
    initial_state = torch.stack((image_t, image_t, image_t, image_t))

    
    state_t = initial_state.clone()
   
    while (True):  # endless running
        loss = 0
        reward_t = 0
        action_t = Action.DO_NOTHING
        # choose an action epsilon gredy
        if t % constant.FRAME_PER_ACTION == 0:  # parameter to skip frames for actions
            action_t = agent.get_action(epsilon,state_t)

        # We reduced the epsilon (exploration parameter) gradually
        if epsilon > constant.FINAL_EPSILON:
            epsilon -= (constant.INITIAL_EPSILON -
                        constant.FINAL_EPSILON) / constant.EXPLORE

        # run the selected action and observed next state and reward
        image_t1, reward_t, terminal = _get_game_state(game, action_t)

        state_t1 = state_t.clone()
        state_t1[0] = state_t[1].clone()
        state_t1[1] = state_t[2].clone()
        state_t1[2] = state_t[3].clone()
        state_t1[3] = image_t1.clone()

        # show(make_grid([state_t[0].cpu(),state_t[1].cpu(),state_t[2].cpu(),state_t[3].cpu(),state_t1[0].cpu(),state_t1[1].cpu(),state_t1[2].cpu(),state_t1[3].cpu()],nrow=4,padding=10))

        # store the transition in experience_replay_memory
        experience_replay_memory.append((state_t, action_t, reward_t, state_t1, terminal))

        memory_len = len(experience_replay_memory)

        if memory_len > constant.REPLAY_MEMORY:
            experience_replay_memory.popleft()

        if(t > constant.OBSERVATION):
            state_batch = random.sample(experience_replay_memory, constant.BATCH)
            loss = agent.learn(state_batch)

            if t % 10 == 0:
                print("t:", t,  "loss:", loss.tolist())

            if t % constant.UPDATETARGETNET == 0:
                agent.update_target_net()

        if terminal:
            state_t = initial_state.clone()
        else:
            state_t = state_t1.clone()

        t = t + 1

def main():
    gpus = [0, 1, 2, 3]
    torch.cuda.set_device(gpus[0])

    agent = DinoAgent(device)  
    game  = Game()

    try:
        train(game,agent)
    except StopIteration:
        game.end()

if __name__ == '__main__':
    main()
