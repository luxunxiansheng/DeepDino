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

import torch

from agents.base_agent import BaseAgent
from game import Game
from utils.logger import Logger
from utils.utilis import Utilis


def main():

    # setup the GPU/CPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(int(Utilis.gpu_id_with_max_memory()))

    # prepare the Log for recording the RL procedure
    cfg = Utilis.config()
      
    working_mode = cfg['GLOBAL'].get('working_mode')
    
    working_agent = BaseAgent.create(cfg)
    game = Game(cfg)    

    try:
        if working_mode == 'train':
            print('******************The Dino is being trained by ' + cfg['GLOBAL'].get('working_agent') + '*************************')
            
            logger = Logger.get_instance()
            logger.create_log(cfg,'.dqn_acc0')
            working_agent.train(game)
        
        elif working_mode == 'replay':
            print('******************The Dino is being replayed*************************')
            working_agent.replay(game)
        else:
            print("working mode is not found. Check the spelling of working mode in config.ini. ")
            
            
    finally:
        print("Something goes wrong!")
        game.end()


if __name__ == '__main__':
    main()
