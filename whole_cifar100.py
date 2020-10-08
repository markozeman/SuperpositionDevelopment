from keras.utils import to_categorical
from dataset_preparation import get_CIFAR_100
from networks import cnn
import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    # to avoid cuDNN error (https://github.com/tensorflow/tensorflow/issues/24496)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    input_size = (32, 32, 3)
    num_of_classes = 100
    num_of_epochs = 100
    batch_size = 50

    X_train, y_train, X_test, y_test = get_CIFAR_100()

    # normalize input images to have values between 0 and 1
    X_train = np.array(X_train).astype(dtype=np.float64)
    X_test = np.array(X_test).astype(dtype=np.float64)
    X_train /= 255
    X_test /= 255
    X_train = X_train.reshape(X_train.shape[0], *input_size)
    X_test = X_test.reshape(X_test.shape[0], *input_size)

    y_train = to_categorical(y_train, num_classes=num_of_classes)  # one-hot encode
    y_test = to_categorical(y_test, num_classes=num_of_classes)  # one-hot encode
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model = cnn(input_size, num_of_classes)

    history = model.fit(X_train, y_train, epochs=num_of_epochs, batch_size=batch_size, validation_split=0.1, verbose=2)
    print(history.history)


