import io
import json
import os
import pickle
import random
import time
from collections import deque
from configparser import ConfigParser
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid

from agents import dino_agent
from game import Game
from utils import torch_helper

# parser config
config_file = "./config.ini"
config = ConfigParser()
config.read(config_file)
 
if torch.cuda.is_available():
   config['DEVICE']['type']= 'cuda'     
   config['DEVICE']['gpu_id']  = str(torch_helper.gpu_id_with_max_memory())
  
else:
   config['DEVICE']['type']= 'cpu'     
    



def main():
       
    working_agent= dino_agent.DinoAgent.create(config)
    game  = Game()

    try:
        working_agent.train(game)
    except StopIteration:
        game.end()

if __name__ == '__main__':
    main()
