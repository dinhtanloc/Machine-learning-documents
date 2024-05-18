import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from skorch import NeuralNetBinaryClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Read data
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]

# Binary encoding of labels
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# Convert to 2D PyTorch tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Define the model
class SonarClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(60, 60)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(60, 60)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(60, 1)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.output(x)
        return x

# create the skorch wrapper
model = NeuralNetBinaryClassifier(
    SonarClassifier,
    criterion=torch.nn.BCEWithLogitsLoss,
    optimizer=torch.optim.Adam,
    lr=0.0001,
    max_epochs=150,
    batch_size=10,
    verbose=False
)

# k-fold
kfold = StratifiedKFold(n_splits=5, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("mean = %.3f; std = %.3f" % (results.mean(), results.std()))
