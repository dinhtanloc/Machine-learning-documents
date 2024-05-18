import torch
a = torch.randn(3,4)
b = torch.nn.functional.pad(a, (1,1,0,2), value=0)
print(a)
print(b)
