import tensorflow as tf

x = tf.Variable(3.6)

with tf.GradientTape() as tape:
    y = x*x

dy = tape.gradient(y, x)
print(dy)
