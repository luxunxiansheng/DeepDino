import base64
import os
from io import BytesIO

import numpy as np
import pandas as pd
import torch
from PIL import Image

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC 
from selenium.webdriver.common.by import By 

import cv2  # opencv
from torchvision import transforms

from constant import Action


class Game:

    _GAME_URL = "http://apps.thecodepost.org/trex/trex.html"
    _CHROME_DRIVER_PATH = "/home/lb/workspace/chromedriver"
    _INIT_SCRIPT = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"

    # get image from canvas
    _GET_BASE64_SCRIPT = "canvasRunner = document.getElementById('runner-canvas'); \
        return canvasRunner.toDataURL().substring(22)"


    _CAPA=DesiredCapabilities.CHROME
    _CAPA["pageLoadStrategy"]="none"       
    
    transform= transforms.Compose([transforms.CenterCrop((150,600)),transforms.Resize((80,80)) ,transforms.Grayscale(),transforms.ToTensor()])

    def __init__(self,actions_df,scores_df):
       
        self._driver = webdriver.Chrome(executable_path=self._CHROME_DRIVER_PATH,desired_capabilities=self._CAPA)
        self._driver.set_window_position(x=-10, y=0)

        self._wait=WebDriverWait(self._driver,20)        
        self._driver.get(self._GAME_URL)

        self._wait.until(EC.presence_of_all_elements_located((By.ID,"socialbutts")))
        

        self._driver.execute_script("Runner.config.ACCELERATION=0")
        self._driver.execute_script(self._INIT_SCRIPT)

        self._actions_df=actions_df
        self._scores_df=scores_df
        
        # display the processed image on screen using openCV, implemented using python coroutine
        self._display = self._show_img()
        self._display.__next__()  # initiliaze the display coroutine

        

    def get_state(self, action):
        # storing actions in a dataframe
        self._actions_df.loc[len(self._actions_df)] = action
        score = self._get_score()
        reward =1
        is_over = False  # game over
        if action==Action.JUMP:
            self._press_up()

        image = self._grab_screen()

        self._display.send(np.asarray(image))  # display the image on screen
        if self._get_crashed():
            # log the score when game is over
            self._scores_df.loc[len(self._scores_df)] = score
            self.restart()
            reward = -1
            is_over = True
        return self.transform(image).cuda(), torch.tensor(reward).cuda(), torch.tensor(is_over).cuda()  # return the Experience tuple

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

    def _show_img(self, graphs=False):
        """
        Show images in new window
        """
        while True:
            screen = (yield)
            window_title = "logs" if graphs else "game_play"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            cv2.resize(screen, (800, 400))
            cv2.imshow(window_title, screen)
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                cv2.destroyAllWindows()
                break

    def _grab_screen(self):
        image_b64 = self._driver.execute_script(self._GET_BASE64_SCRIPT)
        image=Image.open(BytesIO(base64.b64decode(image_b64)))
        return image
