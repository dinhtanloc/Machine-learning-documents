from collections import OrderedDict
import torch.nn as nn

model = nn.Sequential(OrderedDict([
    ('dense1', nn.Linear(764, 100)),
    ('act1', nn.ReLU()),
    ('dense2', nn.Linear(100, 50)),
    ('act2', nn.ReLU()),
    ('output', nn.Linear(50, 10)),
    ('outact', nn.Sigmoid()),
]))
print(model)
