import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten

(trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()

model = Sequential([
          Input(shape=(28,28,1,)),
          Flatten(),
          Dense(units=84, activation="relu"),
          Dense(units=10, activation="softmax"),
      ])

model.summary()

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics="acc")

history = model.fit(x=trainX, y=trainY, batch_size=256, epochs=10,
                    validation_data=(testX, testY))
