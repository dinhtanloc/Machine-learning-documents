import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(100, activation="relu"),
    Dense(100, activation="relu"),
    Dense(10, activation="sigmoid")
])

def batch_generator(image, label, batchsize):
    N = len(image)
    i = 0
    while True:
        yield image[i:i+batchsize], label[i:i+batchsize]
        i = i + batchsize
        if i + batchsize > N:
            i = 0

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics="sparse_categorical_accuracy")
history = model.fit(batch_generator(train_image, train_label, 32),
                    steps_per_epoch=len(train_image)//32,
                    epochs=50, validation_data=(test_image, test_label), verbose=2)
print(model.evaluate(test_image, test_label))

plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.show()
