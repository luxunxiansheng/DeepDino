import torch

x= torch.tensor([3.,4.],requires_grad=True)
y= torch.tensor([2.,5.],requires_grad=True)
z= x*y

print(z)



out=z.mean()

print(out)


out.backward(retain_graph=False)

print(x.grad==None) 

