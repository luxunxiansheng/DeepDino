from enum import Enum

loss_file_path = "/home/lb/workspace/Dino/objects/loss_df.csv"
actions_file_path = "/home/lb/workspace/Dino/objects/actions_df.csv"
q_value_file_path = "/home/lb/workspace/Dino/objects/q_values.csv"
score_file_path = "/home/lb/workspace/Dino/objects/scores_df.csv"
model_file_path = "/home/lb/workspace/Dino/model/dqn.pkl"


GAMMA = 0.99  # decay rate of past observations original 0.99
OBSERVATION = 100.  # timesteps to observe before training
EXPLORE = 100000  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
IMG_ROWS, IMG_COLS = 40,40
img_channels = 4  # We stack 4 frames


class Action(Enum):
    DO_NOTHING=0
    JUMP=1
   
    def __new__(cls,value):
        member=object.__new__(cls)
        member._value_=value
        return member

    def __int__(self):
        return self.value
