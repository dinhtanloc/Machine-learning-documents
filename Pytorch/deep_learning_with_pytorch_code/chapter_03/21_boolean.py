import torch
a = torch.randn(3,4)
print(a)
print(a[:, (a > -1).all(axis=0)])
