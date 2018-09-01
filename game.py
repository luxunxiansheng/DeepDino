import base64
import os
from io import BytesIO

import numpy as np
import pandas as pd
import torch


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC 
from selenium.webdriver.common.by import By 

from PIL import Image

from utils.action import Action



class Game:

    _GAME_URL = "http://apps.thecodepost.org/trex/trex.html"
    _CHROME_DRIVER_PATH = "/home/lb/workspace/chromedriver"
    _INIT_SCRIPT = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"

    # get image from canvas
    _GET_BASE64_SCRIPT = "canvasRunner = document.getElementById('runner-canvas'); \
        return canvasRunner.toDataURL().substring(22)"


    _CAPA=DesiredCapabilities.CHROME
    _CAPA["pageLoadStrategy"]="none"       
    
    

    def __init__(self):
       
        self._driver = webdriver.Chrome(executable_path=self._CHROME_DRIVER_PATH,desired_capabilities=self._CAPA)
        self._driver.set_window_position(x=-10, y=0)

        self._wait=WebDriverWait(self._driver,20)        
        self._driver.get(self._GAME_URL)

        self._wait.until(EC.presence_of_all_elements_located((By.ID,"socialbutts")))
        

        self._driver.execute_script("Runner.config.ACCELERATION=0")
        self._driver.execute_script(self._INIT_SCRIPT)
      
        

    def get_state(self, action):
        reward =1.0
        is_over = False  # game over
        if action==Action.JUMP:
            self._press_up()

        image = self._grab_screen()
   
        if self._get_crashed():
            # log the score when game is over
            self.restart()
            reward = -1.0
            is_over = True
        return image, reward, is_over  # return the Experience tuple

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
        
        if len(score)==0:
            score=0

        return int(score)

    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")

    def end(self):
        self._driver.close()

    
    def _grab_screen(self):
        image_b64 = self._driver.execute_script(self._GET_BASE64_SCRIPT)
        image=Image.open(BytesIO(base64.b64decode(image_b64)))
        return image
