from enum import Enum

class Action(Enum):
    DO_NOTHING=0
    JUMP=1
   
    def __new__(cls,value):
        member=object.__new__(cls)
        member._value_=value
        return member

    def __int__(self):
        return self.value
