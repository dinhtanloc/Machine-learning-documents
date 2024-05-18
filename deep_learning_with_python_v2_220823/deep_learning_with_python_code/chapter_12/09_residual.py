import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Add, \
                                    MaxPool2D, Flatten, Dense
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model

def residual_block(x, filters):
# store the input tensor to be added later as the identity
    identity = x
    x = Conv2D(filters = filters, kernel_size=(3, 3), strides = (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = relu(x)
    x = Conv2D(filters = filters, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([identity, x])
    x = relu(x)

    return x

(trainX, trainY), (testX, testY) = tf.keras.datasets.cifar10.load_data()

input_layer = Input(shape=(32,32,3,))
x = Conv2D(32, (3, 3), padding="same", activation="relu")(input_layer)
x = residual_block(x, 32)
x = Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
x = residual_block(x, 64)
x = Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
x = residual_block(x, 128)
x = Flatten()(x)
x = Dense(units=84, activation="relu")(x)
x = Dense(units=10, activation="softmax")(x)

model = Model(inputs=input_layer, outputs = x)
model.summary()

model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics="acc")

history = model.fit(x=trainX, y=trainY, batch_size=256, epochs=10,
                    validation_data=(testX, testY))
