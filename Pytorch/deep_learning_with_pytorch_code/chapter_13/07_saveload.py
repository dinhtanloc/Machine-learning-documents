import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data into NumPy arrays
data = load_iris()
X, y = data["data"], data["target"]

# convert NumPy array into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

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

# loss metric and optimizer
loss_fn = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# prepare model and training parameters
n_epochs = 100
batch_size = 5
batch_start = torch.arange(0, len(X), batch_size)

# training loop
for epoch in range(n_epochs):
    for start in batch_start:
        # take a batch
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        # forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()

# test for accuracy
y_pred = model(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
print("Accuracy: %.2f" % acc)

# Save model
torch.save(model.state_dict(), "iris-model.pth")

# Create new model and load states
newmodel = Multiclass()
newmodel.load_state_dict(torch.load("iris-model.pth"))

# test with new model using copied tensor
y_pred = newmodel(X_test)
acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
print("Accuracy: %.2f" % acc)
