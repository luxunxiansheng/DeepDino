import os

import numpy as np
import torch


def gpu_id_with_max_memory():
    os.system('nvidia-smi -q -d Memory|grep -A4 GPU|grep Free >tmp')
    memory_available=[int(x.split()[2]) for x in open ('tmp','r').readlines()]
    return np.argmax(memory_available) 


def get_device(config):
    return torch.device(config['DEVICE']['type']+":"+ config['DEVICE']['gpu_id']) if config['DEVICE']['type'] == 'cuda' else torch.device(config['DEVICE']['type'])
