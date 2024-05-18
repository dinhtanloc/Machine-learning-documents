import numpy as np
import torch

polynomial = np.poly1d([1, 2, 3])
N = 20   # number of samples

# Generate random samples roughly between -10 to +10
X = np.random.randn(N,1) * 5
Y = polynomial(X)

# Prepare input as an array of shape (N,3)
XX = np.hstack([X*X, X, np.ones_like(X)])

# Prepare tensors
w = torch.randn(3, 1, requires_grad=True)  # the 3 coefficients
x = torch.tensor(XX, dtype=torch.float32)  # input sample
y = torch.tensor(Y, dtype=torch.float32)   # output sample
optimizer = torch.optim.NAdam([w], lr=0.01)
print(w)

# Run optimizer
for _ in range(1000):
    optimizer.zero_grad()
    y_pred = x @ w
    mse = torch.mean(torch.square(y - y_pred))
    mse.backward()
    optimizer.step()

print(w)
