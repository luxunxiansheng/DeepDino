from enum import Enum


GAMMA = 0.99  # decay rate of past observations original 0.99
OBSERVATION = 150.  # timesteps to observe before training
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 500000  # number of previous transitions to remember
EXPLORE = 100000  # frames over which to anneal epsilon
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 0.00025
IMG_ROWS, IMG_COLS = 84,84
IMG_STACK_SIZE = 4  # We stack 4 frames
ACTION_SPACE=2
UPDATETARGETNET=10000
DOUBLE_DQN=True

class Action(Enum):
    DO_NOTHING=0
    JUMP=1
   
    def __new__(cls,value):
        member=object.__new__(cls)
        member._value_=value
        return member

    def __int__(self):
        return self.value
