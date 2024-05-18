import torch
a = torch.randn(3,3)
b = torch.randn(3,3)
print(a)
print(b)
print(torch.vstack([a,b]))
