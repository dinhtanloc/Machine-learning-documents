import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError

y_true = [1., 0.]
y_pred = [2., 3.]

mae_loss = MeanAbsoluteError()

print(mae_loss(y_true, y_pred).numpy())
