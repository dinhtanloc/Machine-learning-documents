import numpy as np
import torch
from sklearn.model_selection import train_test_split

data = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = data[:, 0:8]
y = data[:, 8]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
