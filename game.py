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


import base64
import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from common.action import Action
from utils.logger import Logger


class Game(object):

    _INIT_SCRIPT = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"

    # get image from canvas
    _GET_BASE64_SCRIPT = "canvasRunner = document.getElementById('runner-canvas'); \
        return canvasRunner.toDataURL().substring(22)"

    _CAPA = DesiredCapabilities.CHROME
    _CAPA["pageLoadStrategy"] = "none"

    def __init__(self, config):

        self._driver = webdriver.Chrome(executable_path=os.path.join(Path(__file__).parent, config['GAME'].get('chrome_driver_path')), desired_capabilities=self._CAPA)
        self._driver.set_window_position(x=-10, y=0)

        self._wait = WebDriverWait(self._driver, 20)
        self._driver.get(config['GAME'].get('game_url'))

        self._wait.until(EC.presence_of_all_elements_located((By.ID, "socialbutts")))

        self._driver.execute_script("Runner.config.ACCELERATION=0")
        self._driver.execute_script(self._INIT_SCRIPT)

    def get_state(self, action):
        score = self._get_score()
        reward = 0.01
        is_over = False  # game over
        if action == Action.JUMP:
            self._press_up()

        image = self._grab_screen()

        if self._get_crashed():
            self.restart()
            reward = -1.0
            is_over = True
        return image, reward, is_over,score  # return the Experience tuple

    def _get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def _get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")

    def _press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def _get_score(self):
        score_array = self._driver.execute_script(
            "return Runner.instance_.distanceMeter.digits")
        # the javascript object is of type array with score in the formate[1,0,0] which is 100.
        score = ''.join(score_array)

        if len(score) == 0:
            score = 0

        return int(score)

    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")

    def end(self):
        self._driver.close()

    def _grab_screen(self):
        image_b64 = self._driver.execute_script(self._GET_BASE64_SCRIPT)
        image = Image.open(BytesIO(base64.b64decode(image_b64)))
        return image
