import numpy as np
from tensorflow import keras

def prepare_x(x):
    # Scale images to the [0, 1] range
    x = x.astype("float32") / 255
    # Scale images to the [0, 1] range
    x = x.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x = np.expand_dims(x, -1)
    return x

def prepare_y(y, num_classes):
    return keras.utils.to_categorical(y, num_classes)

def prepare_data(num_classes):
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # transform x
    x_train = prepare_x(x_train)
    x_test = prepare_x(x_test)
    # transform y
    y_train = prepare_y(y_train, num_classes)
    y_test = prepare_y(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)
