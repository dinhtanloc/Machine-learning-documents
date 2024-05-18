import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# PyTorch classifier
class PimaClassifier(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super().__init__()
        self.layer = nn.Linear(8, 12)
        self.act = activation()
        self.output = nn.Linear(12, 1)
        self.prob = nn.Sigmoid()
        # manually init weights
        init.kaiming_uniform_(self.layer.weight)
        init.kaiming_uniform_(self.output.weight)

    def forward(self, x):
        x = self.act(self.layer(x))
        x = self.prob(self.output(x))
        return x

# create model with skorch
model = NeuralNetClassifier(
    PimaClassifier,
    criterion=nn.BCELoss,
    optimizer=optim.Adamax,
    max_epochs=100,
    batch_size=10,
    verbose=False
)

# define the grid search parameters
param_grid = {
    'module__activation': [nn.Identity, nn.ReLU, nn.ELU, nn.ReLU6,
                           nn.GELU, nn.Softplus, nn.Softsign, nn.Tanh,
                           nn.Sigmoid, nn.Hardsigmoid]
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
