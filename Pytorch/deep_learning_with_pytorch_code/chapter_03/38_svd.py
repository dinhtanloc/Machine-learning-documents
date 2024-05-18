import torch
a = torch.randn(3,4)
print(a)
print(torch.linalg.svd(a))
