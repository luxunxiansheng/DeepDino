import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import datetime
import time


# Bandit is a stateless machine that just reacts to an action of the agent
class Bandit:
    # @k_arm: # of arms
    def __init__(self, k_arm=10,sigma=10):
        self._k_arm = k_arm
        self._sigma=sigma
           
        # real reward for each action , a nomral distribution
        self._q_true = [-1,0,1,1,2,3,4,5.1,4,3]
        self._best_action = np.argmax(self._q_true)
    
    # take an action and return the reward 
    def step(self, action):
        # generate the reward under N(real reward, sigma)
        reward =self._sigma*np.random.randn() +self._q_true[action]
        return reward

    @property
    def k_arm(self):
        return self._k_arm    

    @property
    def best_action(self):
        return self._best_action   
        
    @property
    def true_q_value(self):
        return self._q_true    

class BanditAgent:
    def __init__(self,k_arm,step_size):
        self._k_arm=k_arm
        self._epoch_tried=0
        self._q_estimation= np.zeros(k_arm)
        self._indices = np.arange(k_arm)      
        # # of chosen times for each action
        self._action_counts = np.zeros(k_arm)
        
        self._step_size= step_size
        self._average_reward = 0
       

    def softmax(self):
        return np.exp(self._q_estimation) / np.sum(np.exp(self._q_estimation))

    def act(self):
        action=np.random.choice(self._indices, p=self.softmax())
        self._action_counts[action] += 1
        return action

    def reset(self):
        self._q_estimation= np.zeros(self._k_arm)
        self._action_counts = np.zeros(self._k_arm)
        self._average_reward = 0

    @property
    def average_reward(self):
        return self._average_reward    

    @average_reward.setter
    def average_reward(self,ar):
        self._average_reward=ar   

    def learn(self,action,reward,baseline,use_baseline):
        one_hot = np.zeros(self._k_arm)
        one_hot[action] = 1

        bl=baseline if use_baseline else 0

        self._q_estimation = self._q_estimation + self._step_size * (reward - bl) * (one_hot - self.softmax()) 


def main():
    epochs =5000
    use_baseline = True
    step_size= 0.0002
    the_sigma= 50

    bandit = Bandit(sigma=the_sigma)
    bandit_agent= BanditAgent(bandit.k_arm,step_size)
    
    fig = plt.figure()
      
    q_true_axes=fig.add_subplot(411)
    q_true_axes.set_ylabel('true_q_value')
    q_true_axes.set_xlim([0,bandit.k_arm])
    q_true_axes.plot(bandit.true_q_value)
    
    prob_axes=fig.add_subplot(412)
    best_actions_axes=fig.add_subplot(413)
    average_axes=fig.add_subplot(414)
    
    plt.show(False)
    
    best_aciton_total=np.zeros(epochs)
    best_action_counts= np.zeros(epochs)
    average_reward=np.zeros(epochs)
    epoch_tried=0
    for epoch in tqdm(range(epochs)): 
        prob_axes.clear()
        prob_axes.set_ylabel("Prob.")
        prob_axes.set_xlim([0,bandit.k_arm])
        prob_axes.set_ylim([0,1])
        instant_prob=bandit_agent.softmax()
        prob_axes.bar(range(bandit.k_arm),instant_prob)
        plt.pause(0.1)
          
        epoch_tried += 1
        action= bandit_agent.act()
        reward=bandit.step(action)
       
        # incremental average reward 
        bandit_agent.average_reward =(epoch_tried - 1.0) / epoch_tried *bandit_agent.average_reward + reward / epoch_tried 
        average_reward[epoch]=bandit_agent.average_reward
                        
        if action==bandit.best_action:
            best_action_counts[epoch]=1
        
        best_aciton_total[epoch]=best_action_counts.sum()/epoch_tried

        best_actions_axes.clear()
        best_actions_axes.grid(True)
        best_actions_axes.set_ylabel("Optimal Total#")
        best_actions_axes.set_xlim([0,epochs])

        best_actions_axes.plot(best_aciton_total)
        
        average_axes.clear()
        average_axes.grid(True)
        average_axes.set_ylabel("Ave.Reward#")
        average_axes.set_xlim([0,epochs])
        average_axes.set_ylim([0,10])
        average_axes.plot(average_reward)

        bandit_agent.learn(action,reward,bandit_agent.average_reward,use_baseline)    

    st = datetime.datetime.fromtimestamp(time.time()).strftime('_%Y-%m-%d-%H-%M-%S')    
    plt.savefig('C:/Data/OrNot/workspace/DeepReinforcementLearningPlayground/results/bandit'+st+'.png') 

    

    

if __name__=='__main__':
    main()
