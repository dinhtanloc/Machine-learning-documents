import torch
a = torch.randn(3,3)
b = torch.randn(3,3)
c = torch.concatenate([a, b])
print(c)
print(torch.split(c, 3, dim=0))
