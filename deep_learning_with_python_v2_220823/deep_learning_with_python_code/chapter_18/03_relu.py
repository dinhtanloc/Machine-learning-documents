import tensorflow as tf
from tensorflow.keras.activations import relu

input_array = tf.constant([-1, 0, 1], dtype=tf.float32)
print(relu(input_array))
