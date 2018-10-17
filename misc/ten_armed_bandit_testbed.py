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
 #/  

# Ten-armed bandit is a one-setp MDP in the conext of reinforcment learning. 
# It is helpful to understand the policy gradient method when applying that 
# into the bandit.
# 
# The  implementation is from 
# https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter02/ten_armed_testbed.py 
# by removing other methods and only keeping the policy gradient remained.

#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Artem Oboturov(oboturov@gmail.com)                             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class Bandit:
    # @k_arm: # of arms
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    
    def __init__(self, k_arm=3, initial=0., step_size=0.1):
        self.k = k_arm
        self.step_size = step_size
        self.indices = np.arange(self.k)
        self.time = 0
        self.average_reward = 0
                
        # real reward for each action , a nomral distribution
        self.q_true = np.random.randn(self.k)

               

    def reset(self):
       
        self.instant_reward= np.zeros(self.k)
        # estimation for each action
        self.q_estimation = np.zeros(self.k) 
       
        self.action_prob = np.exp(self.q_estimation) / np.sum(np.exp(self.q_estimation))

        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true)

        self.instant_reward= np.zeros(self.k)

    # get an action for this bandit with softmax policy
    def act(self):
        return np.random.choice(self.indices, p=self.action_prob)
    
    # take an action, update estimation for this action
    def step(self, action):
         
        plt.ion()     
       
        # generate the reward under N(real reward, 1)
        reward = 50*np.random.randn() +self.q_true[action]
        
        self.instant_reward[action]= reward

        self.time += 1
        self.average_reward =(self.time - 1.0) / self.time * self.average_reward + reward / self.time
        self.action_count[action] += 1

        one_hot = np.zeros(self.k)
        one_hot[action] = 1
        baseline = self.average_reward
        
        f3=plt.subplot(413)
        f3.clear()
        f3.plot(self.q_estimation)
        plt.ylabel('Previous Esit.')
        
        #
        self.q_estimation = self.q_estimation + self.step_size * (reward -baseline) * (one_hot - self.action_prob)

        exp_est = np.exp(self.q_estimation)
        self.action_prob = exp_est / np.sum(exp_est)

        f1=plt.subplot(411)
        f1.clear() 
        f1.plot(self.q_true)
        plt.ylabel('q_true')
        
        f2=plt.subplot(412)
        f2.clear()
        f2.plot(self.instant_reward-self.average_reward)
        plt.ylabel('Reward')

        

        f3=plt.subplot(414)
        f3.clear()
        f3.plot(self.q_estimation)
        plt.ylabel('Esti.')

        plt.pause(0.001)
        plt.show()
        

        

        return reward
   

def main():
    runs = 100
    time = 1000
    bandit = Bandit()
    best_action_counts = np.zeros((runs, time))
    rewards = np.zeros(best_action_counts.shape)

    for run in tqdm(range(runs)):
        bandit.reset()
        for time_step in tqdm(range(time)):
            action = bandit.act()
            reward = bandit.step(action)
            rewards[run, time_step] = reward
            if action == bandit.best_action:
                best_action_counts[run,time_step]=1
            
            
    best_action_counts= best_action_counts.mean(axis=0)
            
    plt.plot(best_action_counts)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.savefig('C:/Data/OrNot/workspace/DeepReinforcementLearningPlayground/results/bandit.png')

    plt.close()
    

if __name__ == '__main__':
    main()

    
     

    