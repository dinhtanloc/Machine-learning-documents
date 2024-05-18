import torch
import torch.nn as nn

ce = nn.NLLLoss()

# softmax to apply on dimension 1, i.e. per row
logsoftmax = nn.LogSoftmax(dim=1)

logits = torch.tensor([[-1.90, -0.29, -2.30], [-0.29, -1.90, -2.30]])
pred = logsoftmax(logits)
indices = torch.tensor([1, 0])
print("Cross entropy: %.3f" % ce(pred, indices))
