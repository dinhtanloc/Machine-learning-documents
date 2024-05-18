import tensorflow as tf
from tensorflow.keras.activations import sigmoid

input_array = tf.constant([-1, 0, 1], dtype=tf.float32)
print(sigmoid(input_array))
