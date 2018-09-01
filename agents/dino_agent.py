import random
from collections import deque
from random import randint

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from model.deep_mind_network import DeepMindNetwork
from utils.action import Action
from utils import torch_helper 


class DinoAgent(object):
    def create(config):
        if config['GLOBAL']['agent']=='DQNAgent':
            return DQNAgent(config)
    
    create=staticmethod(create)

class DQNAgent(DinoAgent):
    def __init__(self, config):
        self._config = config
        
        self._device = torch_helper.get_device(config)
        self._image_stack_size = config['DQN'].getint('img_stack_size')
        self._action_space = config['DQN'].getint('action_space')
        self._momentum = config['DQN'].getfloat('momentum')
        self._lr = config['DQN'].getfloat('learning_rate')
        self._img_rows = config['DQN'].getint('img_rows')
        self._img_columns = config['DQN'].getint('img_columns')
        self._batch_size = config['DQN'].getint('batch')
        self._isDQN = config['DQN'].getboolean('dqn') 
        self._gamma = config['DQN'].getfloat('gamma')
        self._final_epsilon  = config['DQN'].getfloat('final_epsilon')
        self._init_epsilon = config['DQN'].getfloat('init_epsilon')
        self._explore = config['DQN'].getint('explore')
        self._replay_memory = config['DQN'].getint('replay_memory')
        self._update_target_interval = config['DQN'].getint('update_target_interval')
        self._frame_per_action = config['DQN'].getint('frame_per_action')
        self._observations = config['DQN'].getint('observations')

        self._policy_net = DeepMindNetwork(input_channels= self._image_stack_size, output_size=self._action_space).to(self._device)
        self._target_net = DeepMindNetwork(input_channels= self._image_stack_size, output_size=self._action_space).to(self._device)

        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()

        self._criterion= nn.SmoothL1Loss()
     
       
        self._optimizer= optim.RMSprop(self._policy_net.parameters(), momentum=self._momentum, lr=self._lr)

    

    def _get_action(self, epsilon, state):
        action_t= Action.DO_NOTHING
        if random.random() <= epsilon:
            action_index= random.randrange(self._action_space)
        else:
            q= self._policy_net(state.to(self._device))
            action_index= torch.argmax(q).tolist()
        if action_index == 0:
            action_t= Action.DO_NOTHING
        else:
            action_t= Action.JUMP
        return action_t



    def _Double_DQN(self, state_t1):
        # predict the q value of the next state with the policy network
        predicted_Q_sa_t1= self._policy_net(state_t1.to(self._device)).detach()
        # get the action which leads to the max q value of the next state
        the_best_action= torch.argmax(predicted_Q_sa_t1)
        # predict the max q value of the next sate with the target network
        the_optimal_q_value_of_next_state = self._target_net(state_t1.to(self._device))[int(the_best_action)].detach()

        return the_optimal_q_value_of_next_state

    def _DQN(self, state_t1):
        predicted_Q_sa_t1= self._target_net(state_t1.to(self._device)).detach()
        the_optimal_q_value_of_next_state = torch.max(predicted_Q_sa_t1)
        return the_optimal_q_value_of_next_state


    
    def _get_game_state(self,game, action):
       
        transform = transforms.Compose([transforms.CenterCrop((150, 600)), 
                                        transforms.Resize((self._img_rows, self._img_columns)), transforms.Grayscale(), transforms.ToTensor()])
        screen_shot, reward, terminal = game.get_state(action)
        return transform(screen_shot), torch.tensor(reward), torch.tensor(terminal)    

    def _learn(self, samples):
       
        q_value=   torch.zeros(self._batch_size, self._action_space).to(self._device)
        td_target= torch.zeros(self._batch_size, self._action_space).to(self._device)

        for i in range(0, self._batch_size):
            state_t= samples[i][0]
            action_t= samples[i][1]
            reward_t= samples[i][2]
            state_t1= samples[i][3]
            terminal= samples[i][4]

            
            the_optimal_q_value_of_next_state= self._DQN(state_t1) if  self._isDQN  else self._Double_DQN(state_t1)

            # td(0)
            td_target[i][int(action_t)]= reward_t if terminal else reward_t + self._gamma * the_optimal_q_value_of_next_state.tolist()

            predicted_Q_sa_t= self._policy_net(state_t.to(self._device))
            q_value[i][int(action_t)]= predicted_Q_sa_t[int(action_t)]

        loss= self._criterion(q_value, td_target)

        self._optimizer.zero_grad()
        loss.backward()

        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()

        return loss

    def _update_target_net(self):
        self._target_net.load_state_dict(self._policy_net.state_dict())


    def train(self,game):
        t = 0
        experience_replay_memory = deque()
        
        epsilon = self._init_epsilon

        # get next step after performing the action
        image_t, _, terminal =self._get_game_state(game, Action.DO_NOTHING)
        initial_state = torch.stack((image_t, image_t, image_t, image_t))

    
        state_t = initial_state.clone()
   
        while (True):  # endless running
            loss = 0
            reward_t = 0
            action_t = Action.DO_NOTHING
            # choose an action epsilon gredy
           
            if t %  self._frame_per_action== 0:  # parameter to skip frames for actions
                action_t =self._get_action(epsilon,state_t)

            # We reduced the epsilon (exploration parameter) gradually
            
            if epsilon > self._final_epsilon:
                
                epsilon -= ( self._init_epsilon-self._final_epsilon) / self._explore 

            # run the selected action and observed next state and reward
            image_t1, reward_t, terminal =self._get_game_state(game, action_t)

            state_t1 = state_t.clone()
            state_t1[0] = state_t[1].clone()
            state_t1[1] = state_t[2].clone()
            state_t1[2] = state_t[3].clone()
            state_t1[3] = image_t1.clone()

        

            # store the transition in experience_replay_memory
            experience_replay_memory.append((state_t, action_t, reward_t, state_t1, terminal))

            memory_len = len(experience_replay_memory)

            
            if memory_len > self._replay_memory:
                experience_replay_memory.popleft()

            
            if(t > self._observations):
                state_batch = random.sample(experience_replay_memory, self._batch_size)
                loss = self._learn(state_batch)

            if t > self._observations and  t % 10 == 0:
                print("t:", t,  "loss:", loss.tolist())

            
            if t %self._update_target_interval == 0:
                self._update_target_net()

            if terminal:
                state_t = initial_state.clone()
            else:
                state_t = state_t1.clone()

            t = t + 1
