import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

# Load MNIST data
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), # required, otherwise MNIST are in PIL format
    #torchvision.transforms.Normalize((0.5,), (0.5,)),
])
train = torchvision.datasets.MNIST('./datafiles/', train=True, download=True, transform=transform)
test = torchvision.datasets.MNIST('./datafiles/', train=True, download=True, transform=transform)

# For manual feed into the model
X_train = train.data.reshape(-1,1,28,28)
y_train = train.targets
X_test = test.data.reshape(-1,1,28,28)
y_test = test.targets

# As iterator for data and target
train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)

# Neural network model
model = nn.Sequential(
    # assume input 1x28x28
    nn.Conv2d(1, 6, kernel_size=(5,5), stride=1, padding=2),
    nn.Tanh(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
    nn.Tanh(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
    nn.Tanh(),
    nn.Flatten(),
    nn.Linear(120, 84),
    nn.Tanh(),
    nn.Linear(84, 10),
    nn.LogSoftmax(dim=1)
)

# self-defined training loop function
def training_loop(model, optimizer, loss_fn, train_loader, val_loader=None, n_epochs=100):
    best_loss, best_epoch = np.inf, -1
    best_state = model.state_dict()

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        for data, target in train_loader:
            output = model(data)
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # Validation
        model.eval()
        status = (f"{str(datetime.datetime.now())} End of epoch {epoch}, "
                  f"training loss={train_loss/len(train_loader)}")
        if val_loader:
            val_loss = 0
            for data, target in val_loader:
                output = model(data)
                loss = loss_fn(output, target)
                val_loss += loss.item()
            status += f", validation loss={val_loss/len(val_loader)}"
        print(status)

optimizer = optim.Adam(model.parameters())
criterion = nn.NLLLoss()
training_loop(model, optimizer, criterion, train_loader, test_loader, n_epochs=100)
