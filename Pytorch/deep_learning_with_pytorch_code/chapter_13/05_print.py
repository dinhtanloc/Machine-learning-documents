import pprint
import torch.nn as nn

# PyTorch model
class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.output = nn.Linear(8, 3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.logsoftmax(self.output(x))
        return x

model = Multiclass()

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(model.state_dict())
