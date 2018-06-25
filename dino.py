
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


import constant
from constant import Action
from game import Game
from deep_q_network import Deep_Q__network

# Intialize log structures from file if exists else create new
loss_df = pd.read_csv(constant.loss_file_path) if os.path.isfile(
    constant.loss_file_path) else pd.DataFrame(columns=['loss'])
q_values_df = pd.read_csv(constant.q_value_file_path) if os.path.isfile(
    constant.q_value_file_path) else pd.DataFrame(columns=['qvalues'])
actions_df = pd.read_csv() if os.path.isfile(
    constant.actions_file_path) else pd.DataFrame(columns=['actions'])
scores_df = pd.read_csv(constant.score_file_path) if os.path.isfile(
    constant.score_file_path) else pd.DataFrame(columns=['scores'])


def save_obj(obj, name):
    # dump files into objects folder
    with open('/home/lb/workspace/Dino/objects/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('/home/lb/workspace/Dino/objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_weights(network):
    with open(constant.model_file_path, 'wb') as f:  # dump files into objects folder
        torch.save(network.state_dict(), f)


def load_weights(network):
    network.load_state_dict(torch.load(constant.model_file_path))


# training variables saved as checkpoints to filesystem to resume training from the same step


def init_cache(q_value_network):
    """initial variable caching, done only once"""
    save_obj(constant.INITIAL_EPSILON, "epsilon")
    t = 0
    save_obj(t, "time")
    experience_replay_memory = deque()
    save_obj(experience_replay_memory, "experience_replay_memory")
    save_weights(q_value_network)


def learn(samples):
    
    q_value= torch.zeros(constant.BATCH)
    td_target=torch.zeros([constant.BATCH])
        
    for i in range(0,constant.BATCH):
        state_t  = samples[i][0]
        action_t = samples[i][1]
        reward_t = samples[i][2]
        state_t1 = samples[i][3]
        terminal = samples[i][4]

                        
            # td(0) 
        Q_sa_t1 = self.forward(state_t1) 
        td_target[i] = reward_t if terminal else reward_t+ constant.GAMMA * torch.max(Q_sa_t1).tolist()
         

        Q_sa_t = self.forward(state_t)
        q_value[i]   = Q_sa_t[int(action_t)]
  
            
    loss= self.criterion(q_value,td_target).cuda()
                
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss 




def train(game, deep_q_network):

    last_time = time.time()
    # restore the previous
    experience_replay_memory = load_obj("experience_replay_memory")  # load from file system
    load_weights(deep_q_network)
    epsilon = load_obj("epsilon")
    t = load_obj("time")
    Q_sa_t1 = torch.zeros([2])

    # get next step after performing the action
    image_t, _, terminal = game.get_state(Action.DO_NOTHING)
    initial_state = torch.stack((image_t, image_t, image_t, image_t)).view(1, 4, 80, 80)
    state_t1 = initial_state
    state_t = initial_state
  
    
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

        # helpful for measuring frame rate
        print('fps: {0}'.format(1 / (time.time()-last_time)))
        last_time = time.time()

        state_t1[0][0] = state_t[0][1]
        state_t1[0][1] = state_t[0][2]
        state_t1[0][2] = state_t[0][3]
        state_t1[0][3] = image_t1

        # store the transition in experience_replay_memory
        experience_replay_memory.append((state_t, action_t, reward_t, state_t1, terminal))

        memory_len = len(experience_replay_memory)

        if memory_len > constant.REPLAY_MEMORY:
            experience_replay_memory.popleft()


        if(t>constant.OBSERVATION):

            state_batch = random.sample(experience_replay_memory, constant.BATCH)
            loss=deep_q_network.learn(state_batch)
            loss_df.loc[len(loss_df)] = loss.tolist()
            #q_values_df.loc[len(q_values_df)] = torch.max(Q_sa_t1, 1)[0][0].tolist()
        
        # reset game to initial frame if terminate
        state_t = initial_state if terminal else state_t1
        t = t + 1

        # save progress every 1000 iterations
        if t % 1000 == 0:
            print("Now we save model")
            game.pause()  # pause game while saving to filesystem
            save_weights(deep_q_network)
            save_obj(experience_replay_memory,"experience_replay_memory")  # saving episodes
            save_obj(t, "time")  # caching time steps
            # cache epsilon to avoid repeated randomness in actions
            save_obj(epsilon, "epsilon")
            loss_df.to_csv(constant.loss_file_path, index=False)
            scores_df.to_csv(constant.score_file_path, index=False)
            actions_df.to_csv(constant.actions_file_path, index=False)
            q_values_df.to_csv(constant.q_value_file_path, index=False)
            game.resume()
        # print info
        state = ""
        if t <= constant.OBSERVATION:
            state = "observe"
        elif t <= constant.OBSERVATION + constant.EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION",
              action_t, "/ REWARD", reward_t.tolist(), "/ Q_MAX ", torch.max(Q_sa_t1).tolist(), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")




def e_greedy(epsilon, q_value_network, state):

    action_t = Action.DO_NOTHING

    if random.random() <= epsilon:  # randomly explore an action
        print("----------Random Action----------")
        action_index = random.randrange(len(Action))

    else:  
        print("----------Greedy Action----------")
        q = q_value_network(state)
        action_index = torch.argmax(q).tolist()
    
    if action_index == 0:
        action_t = Action.DO_NOTHING
    else:
        action_t = Action.JUMP   
    return action_t


def main():

    gpus = [0, 1, 2, 3]
    torch.cuda.set_device(gpus[3])

    DQN = Deep_Q__network()
    DQN.cuda()

    init_cache(DQN)
    game = Game(actions_df, scores_df)

    try:
        train(game, DQN)
    except StopIteration:
        game.end()


if __name__ == '__main__':
    main()
