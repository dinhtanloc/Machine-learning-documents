import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

(train_image, train_label), (test_image, test_label) = load_data()
print(train_image.shape)
print(train_label.shape)
print(test_image.shape)
print(test_label.shape)

model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(100, activation="relu"),
    Dense(100, activation="relu"),
    Dense(10, activation="sigmoid")
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics="sparse_categorical_accuracy")
history = model.fit(train_image, train_label,
                    batch_size=32, epochs=50,
                    validation_data=(test_image, test_label), verbose=0)
print(model.evaluate(test_image, test_label))

plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.show()
