import torch

x = torch.tensor(3.6, requires_grad=True)
y = x * x
y.backward()
print("x =", x)
print("y =", y)
print("x.grad =", x.grad)
