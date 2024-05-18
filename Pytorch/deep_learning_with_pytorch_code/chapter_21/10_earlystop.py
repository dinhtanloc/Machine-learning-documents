import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split, default_collate
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

data = fetch_openml("electricity", version=1, parser="auto")

# Label encode the target, convert to float tensors
X = data['data'].astype('float').values
y = data['target']
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# train-test split for model evaluation
trainset, testset = random_split(TensorDataset(X, y), [0.7, 0.3])

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    model.load_state_dict(torch.load(filename))

# Define the model
model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 12),
    nn.ReLU(),
    nn.Linear(12, 1),
    nn.Sigmoid(),
)

# Train the model
n_epochs = 10000  # more than we needed
loader = DataLoader(trainset, shuffle=True, batch_size=32)
X_test, y_test = default_collate(testset)
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

early_stop_thresh = 5
best_accuracy = -1
best_epoch = -1

for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    acc = float(acc) * 100
    print(f"End of epoch {epoch}: accuracy = {acc:.2f}%")
    if acc > best_accuracy:
        best_accuracy = acc
        best_epoch = epoch
        checkpoint(model, "best_model.pth")
    elif epoch - best_epoch > early_stop_thresh:
        print("Early stopped training at epoch %d" % epoch)
        break  # terminate the training loop

resume(model, "best_model.pth")
