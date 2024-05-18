import torch
a = torch.randn(3,4)
print(a)
print(torch.mean(a, dim=0))
print(torch.std(a, dim=0))
print(torch.cumsum(a, dim=0))
print(torch.cumprod(a, dim=0))
