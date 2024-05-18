import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

y_true = [1., 0.]
y_pred = [2., 3.]

mse_loss = MeanSquaredError()

print(mse_loss(y_true, y_pred).numpy())
