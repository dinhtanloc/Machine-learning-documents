import torch.nn as nn

# PyTorch classifier
class PimaClassifier(nn.Module):
    def __init__(self, weight_init=nn.init.xavier_uniform_):
        super().__init__()
        self.layer = nn.Linear(8, 12)
        self.act = nn.ReLU()
        self.output = nn.Linear(12, 1)
        self.prob = nn.Sigmoid()
        # manually init weights
        weight_init(self.layer.weight)
        weight_init(self.output.weight)

    def forward(self, x):
        x = self.act(self.layer(x))
        x = self.prob(self.output(x))
        return x
