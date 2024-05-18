import torch

target = torch.tensor([[0., 1., 0.], [1., 0., 0.]])
indices = torch.argmax(target, dim=1)
print(indices)
