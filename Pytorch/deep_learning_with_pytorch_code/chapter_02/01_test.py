import torch

# create a tensor of size (2,3)
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# create a tensor of size (3,2)
y = torch.tensor([[1, 2], [3, 4], [5, 6]])

# matrix multiplication of x and y
z = torch.matmul(x, y)

print(z)
