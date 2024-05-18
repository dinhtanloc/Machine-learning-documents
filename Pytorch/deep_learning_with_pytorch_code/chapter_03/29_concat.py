import torch
a = torch.randn(3,3)
b = torch.randn(3,3)
c = torch.concatenate([a, b])
print(a)
print(b)
print(c)
