import torch
import torch.nn as nn

bce = nn.BCELoss()

pred = torch.tensor([0.75, 0.25])
target = torch.tensor([1., 0.])
print("Binary cross entropy: %.3f" % bce(pred, target))
