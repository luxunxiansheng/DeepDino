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


import os
import shutil
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
    
import errno


class Utilis(object):
    @staticmethod
    def gpu_id_with_max_memory():
        os.system('nvidia-smi -q -d Memory|grep -A4 GPU|grep Free > dump')
        memory_available = [int(x.split()[2]) for x in open('dump', 'r').readlines()]
        os.system('rm ./dump')
        return np.argmax(memory_available)

    @staticmethod
    def config():
        # parser config
        config_file = "config.ini"
        config = ConfigParser()
        config.read(os.path.join(Path(__file__).parents[1], config_file))
        return config

    @staticmethod
    def layer_init(layer, w_scale=1.0):
        nn.init.orthogonal_(layer.weight.data)
        layer.weight.data.mul_(w_scale)
        nn.init.constant_(layer.bias.data, 0)
        return layer
    



    # Taken from https://stackoverflow.com/a/600612/119527
    @staticmethod
    def mkdir_p(path):
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise
    
    @staticmethod
    def safe_open_w(path):
        ''' Open "path" for writing, creating any parent directories as needed.'''
        Utilis.mkdir_p(os.path.dirname(path))
        return open(path, 'w')

        

    @staticmethod
    def save_checkpoint(checkpoint, is_best, agentname,checkpointpathname='checkpoint'):
        checkpoint_file_path=  os.path.join(Path(__file__).parents[1],checkpointpathname,agentname)    
        Utilis.mkdir_p(checkpoint_file_path)

        checkfile=os.path.join(checkpoint_file_path,'checkpoint.pth.tar')
                
        torch.save(checkpoint, checkfile)
        if is_best:
            best_file=os.path.join(checkpoint_file_path,'model_best.pth.tar')
            shutil.copyfile(checkfile,best_file)
    
    @staticmethod
    def load_checkpoint(agentname, checkpointpathname='checkpoint'):
        checkpoint_file=  os.path.join(Path(__file__).parents[1],checkpointpathname,agentname,'checkpoint.pth.tar')    
        if os.path.isfile(checkpoint_file):
           return torch.load(checkpoint_file)
        else:
            return None
    
    @staticmethod
    def load_best_checkpoint(agentname, checkpointpathname='checkpoint'):    
        model_best_file = os.path.join(Path(__file__).parents[1], checkpointpathname, agentname, 'model_best.pth.tar')
        if os.path.isfile(model_best_file):
            return torch.load(model_best_file)
        else:
            return None