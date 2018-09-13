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
import time
from pathlib import Path
from collections import namedtuple


import pandas as pd




class Logger(object):

    LOG_FILE_HEADER=['Time_Step','Episode','Score','Loss']
    LOG_ENTRY= namedtuple("Log_Entry",'time_step episode score epoch_loss')
    
    _instance = None

    @staticmethod
    def get_instance():
        if Logger._instance == None:
            Logger()

        return Logger._instance

    def create_log_file(self, config):  # Explictly init the logger anyhow
        
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%Y%m%d")
        timestampLaunch = timestampDate + '-' + timestampTime

        project_root_dir=Path(__file__).parents[1]
        self._log_file = os.path.join(project_root_dir, config["GAME"].get("game_log_file_path"), timestampLaunch + "-game.csv")
        
        
    def __init__(self):
        if Logger._instance != None:
            raise Exception("Logger is a Singleon")
        else:
            Logger._instance = self

        self._game_log = pd.DataFrame(columns=Logger.LOG_FILE_HEADER)    

    def log_game(self, log_entry):
         self._game_log=self._game_log.append({'Time_Step': int(log_entry.time_step), 'Episode': int(log_entry.episode), 'Score': int(log_entry.score), 'Loss': log_entry.epoch_loss}, ignore_index=True)
        
    

    def dump_log(self):
        if self._log_file!=None:
            self._game_log.to_csv(self._log_file, index=False)
        else:
            raise Exception("No Log files exist")

    
