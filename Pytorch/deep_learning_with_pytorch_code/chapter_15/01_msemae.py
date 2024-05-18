import torch
import torch.nn as nn

mae = nn.L1Loss()
mse = nn.MSELoss()

predict = torch.tensor([0., 3.])
target = torch.tensor([1., 0.])

print("MAE: %.3f" % mae(predict, target))
print("MSE: %.3f" % mse(predict, target))
