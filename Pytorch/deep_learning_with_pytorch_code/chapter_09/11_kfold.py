import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import StratifiedKFold

data = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = data[:, 0:8]
y = data[:, 8]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

def model_train(X_train, y_train, X_test, y_test):
    # create new model
    model = nn.Sequential(
        nn.Linear(8, 12),
        nn.ReLU(),
        nn.Linear(12, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )

    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 25    # number of epochs to run
    batch_size = 10  # size of each batch
    batches_per_epoch = len(X_train) // batch_size

    for epoch in range(n_epochs):
        with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0, disable=True
                        ) as bar:
            bar.set_description(f"Epoch {epoch}")
            for i in bar:
                # take a batch
                start = i * batch_size
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
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
    # evaluate accuracy at end of training
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    return float(acc)

# define 5-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cv_scores = []
for train, test in kfold.split(X, y):
    # create model, train, and get accuracy
    acc = model_train(X[train], y[train], X[test], y[test])
    print("Accuracy: %.2f" % acc)
    cv_scores.append(acc)
# evaluate the model
print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores)*100, np.std(cv_scores)*100))
