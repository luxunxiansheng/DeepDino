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


import pandas as pd


class Logger(object):

    _instance = None

    @staticmethod
    def get_instance():
        if Logger._instance == None:
            Logger()

        return Logger._instance

    def init(self, config):  # Explictly init the logger anyhow
        
        timestampTime = time.strftime("%H%M%S")
        timestampDate = time.strftime("%Y%m%d")
        timestampLaunch = timestampDate + '-' + timestampTime


        project_root_dir=Path(__file__).parents[1]
        self._score_file = os.path.join(project_root_dir,config["GAME"].get("score_log_file_path"),timestampLaunch+"-score.csv")
        self._score_log = pd.read_csv(self._score_file) if os.path.isfile(self._score_file) else pd.DataFrame(columns=['scores'])

        self._q_value_file =os.path.join(project_root_dir,config['DQN'].get('q_value_log_file_path'),timestampLaunch+"-qvalue.csv")
        self._q_values_log = pd.read_csv(self._q_value_file) if os.path.isfile(self._q_value_file) else pd.DataFrame(columns=['qvalues'])

    def __init__(self):
        if Logger._instance != None:
            raise Exception("Logger is a Singleon")
        else:
            Logger._instance = self

    def log_game_score(self, score):
        self._score_log.loc[len(self._score_log)] = score

    def dump_log(self):
        self._score_log.to_csv(self._score_file, index=False)
        self._q_values_log.to_csv(self._q_value_file, index=False)
