import torch

t1 = torch.tensor([1, 1])
t2 = torch.tensor([2, 2])
t3 = torch.tensor([3, 3])
t4 = torch.tensor([4, 4])
t5 = torch.tensor([5, 5])

init = torch.stack([t1, t2,t3])

t_now = init
print(t_now)



t_next=t_now.clone()
print(t_next)



t_next[0:-1] = t_now[1:]
print(t_next)

t_next[-1] = t4

print(t_next)




print(t_now)
print(t_next)


t2 = torch.tensor([8, 9])

print(t_now)
print(t_next)













