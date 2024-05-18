import torch
a = torch.zeros(2, 3, 4)
b = a.type(torch.int32)
print(a.dtype)
print(b.dtype)
