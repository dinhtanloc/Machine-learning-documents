import torch
import torch.nn as nn

ce = nn.CrossEntropyLoss()

logits = torch.tensor([[-1.90, -0.29, -2.30], [-0.29, -1.90, -2.30]])
indices = torch.tensor([1, 0])
print("Cross entropy: %.3f" % ce(logits, indices))
