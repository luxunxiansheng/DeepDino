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
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import torch


class Utilis(object):

    @staticmethod
    def gpu_id_with_max_memory():
        os.system('nvidia-smi -q -d Memory|grep -A4 GPU|grep Free > dump')
        memory_available = [int(x.split()[2]) for x in open('dump', 'r').readlines()]

        os.system('rm ./dump')
        return np.argmax(memory_available)

    @staticmethod
    def get_device(config):
        return torch.device(config['DEVICE']['type']+":" + config['DEVICE']['gpu_id']) if config['DEVICE']['type'] == 'cuda' else torch.device(config['DEVICE']['type'])

    @staticmethod
    def config():
        # parser config
        config_file = "config.ini"

        config = ConfigParser()

        config.read(os.path.join(Path(__file__).parents[1], config_file))

        # setup the GPU/CPU device
        if torch.cuda.is_available():
            config['DEVICE']['type'] = 'cuda'
            config['DEVICE']['gpu_id'] = str(Utilis.gpu_id_with_max_memory())

        else:
            config['DEVICE']['type'] = 'cpu'

        return config
