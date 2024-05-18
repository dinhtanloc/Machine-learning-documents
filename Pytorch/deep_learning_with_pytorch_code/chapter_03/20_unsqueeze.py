import torch
a = torch.randn(3,4,5)
b = torch.unsqueeze(a, dim=2)
print(a.shape)
print(b.shape)
