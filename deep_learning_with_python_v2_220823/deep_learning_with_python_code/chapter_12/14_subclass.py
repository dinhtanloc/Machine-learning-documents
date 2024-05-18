import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.models import Model

class LeNet5(tf.keras.Model):
  def __init__(self):
    super(LeNet5, self).__init__()
    #creating layers in initializer
    self.conv1 = Conv2D(6, (5,5), padding="same", activation="relu")
    self.max_pool2x2 = MaxPool2D(pool_size=(2,2))
    self.conv2 = Conv2D(16, (5,5), padding="same", activation="relu")
    self.conv3 = Conv2D(120, (5,5), padding="same", activation="relu")
    self.flatten = Flatten()
    self.fc2 = Dense(units=84, activation="relu")
    self.fc3=Dense(units=10, activation="softmax")

  def call(self, input_tensor):
    # don't add layers here, need to create the layers in initializer,
    # otherwise you will get the tf.Variable can only be created once error
    x = self.conv1(input_tensor)
    x = self.max_pool2x2(x)
    x = self.conv2(x)
    x = self.max_pool2x2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x

input_layer = Input(shape=(32,32,3,))
x = LeNet5()(input_layer)
model = Model(inputs=input_layer, outputs=x)
model.summary(expand_nested=True)
