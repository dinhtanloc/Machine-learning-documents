import torch
a = torch.randn(2, 3)
b = torch.randn(2, 3)
print(torch.matmul(a, b.T))
print(a @ b.T)
