import tensorflow as tf
import numpy as np

n_dataset = tf.data.Dataset.from_tensor_slices(np.arange(10000))
shuffled = []
for n in n_dataset.shuffle(10).take(20):
    shuffled.append(n.numpy())
print(shuffled)
