from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPool2D

model = Sequential([
          Input(shape=(32,32,3,)),
          Conv2D(6, (5,5), padding="same", activation="relu"),
          MaxPool2D(pool_size=(2,2)),
          Conv2D(16, (5,5), padding="same", activation="relu"),
          MaxPool2D(pool_size=(2, 2)),
          Conv2D(120, (5,5), padding="same", activation="relu"),
          Flatten(),
          Dense(units=84, activation="relu"),
          Dense(units=10, activation="softmax"),
      ])

model.summary()
