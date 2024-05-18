import torch
a = torch.randn(3)
b = torch.randn(3)
print(a)
print(b)
print(torch.dot(a, b))
print(a @ b)
