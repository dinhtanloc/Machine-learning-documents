import numpy as np

polynomial = np.poly1d([1, 2, 3])
N = 20   # number of samples

# Generate random samples roughly between -10 to +10
X = np.random.randn(N,1) * 5
Y = polynomial(X)
print(X)
print(Y)
