from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten

model = Sequential([
          Input(shape=(28,28,1,)),
          Flatten(),
          Dense(units=84, activation="relu"),
          Dense(units=10, activation="softmax"),
      ])

print (model.summary())
