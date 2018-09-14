# DeepDino
A carton game powered by deep reinforcement  learning algorithm 

A playground to try various RL algorithms 




# Programming Philosophy 
Python is a very flexible programing language and can be easilly used in a 
procedure-oriented way. In fact, procedure-oriented-design is what many 
machine learning algorithms are being flollowed. 

But I dislike it. After reading many guys' codes, I just feel not good. 
ML algortim is somehow complexy in itself, but if the code is not writen 
caerfully, it is even a double nigthmare. To make life easier, in this 
project, I hope the code is at least readable in the first palce , and
the algorithm efficiency is secondary. So I decided to follow Object-
Oriented-Design as much as possible for I believe OOD is mostly  consistent
with the human intuition. In the last decades, numerous great softwares from 
industry were designed and programmed with OO principle. If used properly,
OO is a swiss knife without any doubt. 


# Visualization 
To monitor the training process and the visual the results , we take advantage of 
the tensorboard as the toolkit.More specifically, we adopt the code posted in repo: 
"https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard" 
where the log entry is wrapped accroding to tensorflow Summary format as to be read by 
tensorboard. 

To use the tensorboad, just type the command in a terminal as below:

*
*    tensorboad --logdir ./log  --port 8008 
*

A tensorboard server will be launched and you can visit the tensorflow web page from a 
browser just as promoted in the terminal



