import torch.nn as nn
from skorch import NeuralNetClassifier

class SonarClassifier(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.layers = []
        self.acts = []
        for i in range(n_layers):
            self.layers.append(nn.Linear(60, 60))
            self.acts.append(nn.ReLU())
            self.add_module(f"layer{i}", self.layers[-1])
            self.add_module(f"act{i}", self.acts[-1])
        self.output = nn.Linear(60, 1)

    def forward(self, x):
        for layer, act in zip(self.layers, self.acts):
            x = act(layer(x))
        x = self.output(x)
        return x

model = NeuralNetClassifier(
    module=SonarClassifier,
    max_epochs=150,
    batch_size=10,
    module__n_layers=2
)
