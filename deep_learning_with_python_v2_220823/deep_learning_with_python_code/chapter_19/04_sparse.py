import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

y_true = [1, 0]
y_pred = [[0.15, 0.75, 0.1], [0.75, 0.15, 0.1]]

cross_entropy_loss = SparseCategoricalCrossentropy()

print(cross_entropy_loss(y_true, y_pred).numpy())
