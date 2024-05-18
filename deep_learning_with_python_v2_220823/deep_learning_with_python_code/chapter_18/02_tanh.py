import tensorflow as tf
from tensorflow.keras.activations import tanh

input_array = tf.constant([-1, 0, 1], dtype=tf.float32)
print(tanh(input_array))
