import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from everywhereml.code_generators.tensorflow import tf_porter

def get_model():
    x_values = np.random.uniform(low=0, high=2 * math.pi, size=10000)
    y_values = np.sin(x_values)
    x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.3)
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.3)

    # create a NN with 2 layers of 16 neurons
    model = tf.keras.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(1,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_data=(x_validate, y_validate))

    return model, x_train, y_train

tf_model, x_train, y_train = get_model()
# tf_porter() requires:
#   1. the neural network model
#   2. the input data (to detect the input dimensions)
#   3. the output labels (to detect the number of classes - if classification)
#
# Passing `instance_name` will create an instance of the model, so you don't have to
# `area_size` is to control how much memory to allocate for the network
# It is a trial-and-error process
porter = tf_porter(tf_model, x_train, y_train)
cpp_code = porter.to_cpp(instance_name='sineNN', arena_size=4096)

print(cpp_code)