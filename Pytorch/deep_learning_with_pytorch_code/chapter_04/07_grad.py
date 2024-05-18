import numpy as np
import torch

polynomial = np.poly1d([1, 2, 3])

# Generate random samples roughly between -10 to +10
N = 20   # number of samples
X = np.random.randn(N,1) * 5
Y = polynomial(X)

XX = np.hstack([X*X, X, np.ones_like(X)])

w = torch.randn(3, 1, requires_grad=True)  # the 3 coefficients
x = torch.tensor(XX, dtype=torch.float32)  # input sample
y = torch.tensor(Y, dtype=torch.float32)   # output sample
optimizer = torch.optim.NAdam([w], lr=0.01)
print(w)

for _ in range(1000):
    y_pred = x @ w
    mse = torch.mean(torch.square(y - y_pred))
    optimizer.zero_grad()
    mse.backward()
    optimizer.step()

print(w)
