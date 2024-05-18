import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, Flatten, \
                                    Conv2D, BatchNormalization, MaxPool2D
from tensorflow.keras.models import Model

(trainX, trainY), (testX, testY) = keras.datasets.cifar10.load_data()

input_layer = Input(shape=(32,32,3,))
x = Conv2D(6, (5,5), padding="same", activation="relu")(input_layer)
x = MaxPool2D(pool_size=(2,2))(x)
x = Conv2D(16, (5,5), padding="same", activation="relu")(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Conv2D(120, (5,5), padding="same", activation="relu")(x)
x = Flatten()(x)
x = Dense(units=84, activation="relu")(x)
x = Dense(units=10, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=x)
model.summary()

model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics="acc")

history = model.fit(x=trainX, y=trainY, batch_size=256, epochs=10,
                    validation_data=(testX, testY))
