import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy

# using one-hot vector representation
y_true = [[0, 1, 0], [1, 0, 0]]
y_pred = [[0.15, 0.75, 0.1], [0.75, 0.15, 0.1]]

cross_entropy_loss = CategoricalCrossentropy()

print(cross_entropy_loss(y_true, y_pred).numpy())
