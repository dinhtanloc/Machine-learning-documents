import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but Tensorflow will figure that out for us.)
W = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.zeros([1]))

# A function to compute mean squared error between y_data and computed y
def mse_loss():
    y = W * x_data + b
    loss = tf.reduce_mean(tf.square(y - y_data))
    return loss

# Minimize the mean squared errors.
optimizer = tf.keras.optimizers.Adam()
for step in range(5000):
    optimizer.minimize(mse_loss, var_list=[W,b])
    if step % 500 == 0:
        print(step, W.numpy(), b.numpy())

# Learns best fit is W: [0.1], b: [0.3]
