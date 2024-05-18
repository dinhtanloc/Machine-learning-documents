import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skorch import NeuralNetBinaryClassifier

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

model = NeuralNetBinaryClassifier(
    SonarClassifier,
    criterion=torch.nn.BCEWithLogitsLoss,
    optimizer=torch.optim.Adam,
    lr=0.0001,
    max_epochs=150,
    batch_size=10,
    verbose=False
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('float32', FunctionTransformer(func=lambda X: torch.tensor(X, dtype=torch.float32),
                                    validate=False)),
    ('sonarmodel', model.initialize()),
])

param_grid = {
    'sonarmodel__module__n_layers': [1, 3, 5],
    'sonarmodel__lr': [0.1, 0.01, 0.001, 0.0001],
    'sonarmodel__max_epochs': [100, 150],
}

grid_search = GridSearchCV(pipe, param_grid, scoring='accuracy', verbose=1, cv=3)
result = grid_search.fit(X, y)
print("Best: %f using %s" % (result.best_score_, result.best_params_))
means = result.cv_results_['mean_test_score']
stds = result.cv_results_['std_test_score']
params = result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
