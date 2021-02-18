# from keras import Sequential
# from keras.engine import InputLayer
# from keras.layers import Dense, Flatten
# from keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from preprocess import preprocess_SEA, preprocess_Stagger
from help_functions import random_binary_array, context_stats, get_context_matrices
from plots import plot_general
from superposition import superposition_training_cifar, normal_training_cifar
import numpy as np


def nn(input_size, num_of_units, num_of_classes):
    """
    Create simple NN model with two hidden layers, each has 'num_of_units' neurons.

    :param input_size: image input size in pixels
    :param num_of_units: number of neurons in each hidden layer
    :param num_of_classes: number of different classes/output labels
    :return: Keras model instance
    """
    model = Sequential()
    model.add(Flatten(input_shape=(1, input_size)))     # Flatten added for compatibility with superposition code
    model.add(Dense(num_of_units, activation='relu'))
    model.add(Dense(num_of_units, activation='relu'))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def training(model, X, y, input_size, num_of_units, num_of_epochs, num_of_tasks, batch_size):
    """
    Train SEA concept or Stagger data in superposition.
    Make separate test data on the first task and measure test accuracy on it over time.

    :param model: Keras model instance
    :param X: feature input data
    :param y: output labels
    :param input_size:
    :param num_of_units:
    :param num_of_epochs:
    :param num_of_tasks: tells into how many equal sized tasks X and y are split
    :param batch_size:
    :return:
    """
    nn_cnn = 'nn'

    X_split = np.array_split(X, num_of_tasks)
    y_split = np.array_split(y, num_of_tasks)

    d = []
    for X_s, y_s in zip(X_split, y_split):
        # 1.000 samples for test data
        X_train = X_s[1000:, :]
        X_test = X_s[:1000, :]
        y_train = y_s[1000:, :]
        y_test = y_s[:1000, :]
        d.append((X_train, y_train, X_test, y_test))

    context_matrices = get_context_matrices((input_size, 1), num_of_units, num_of_tasks)
    acc_superposition = superposition_training_cifar(model, d, num_of_epochs, num_of_tasks, context_matrices, nn_cnn, batch_size)

    model = nn(input_size, num_of_units, num_of_classes)    # new model for normal training
    acc_normal = normal_training_cifar(model, d, num_of_epochs, num_of_tasks, nn_cnn, batch_size)

    plot_general(acc_superposition, acc_normal, ['Superposition model', 'Baseline model'],
                 'Superposition vs. baseline model with ' + nn_cnn.upper() + ' model', 'Epoch', 'Accuracy (%)',
                 [(i + 1) * num_of_epochs for i in range(num_of_tasks - 1)], 0, 100)


if __name__ == '__main__':
    num_of_tasks = 3
    num_of_epochs = 10
    batch_size = 32

    # sea_path = 'data/SEA concept/SEA_60k.data'
    # X_sea, y_sea = preprocess_SEA(sea_path)
    #
    # # X_sea = X_sea[:15000, :]
    # # y_sea = y_sea[:15000, :]
    #
    # input_size = 3
    # num_of_units = 10
    # num_of_classes = 2
    # model = nn(input_size, num_of_units, num_of_classes)
    #
    # # model.fit(X_sea, y_sea, epochs=num_of_epochs, batch_size=batch_size, verbose=2, validation_split=0.1)
    #
    # training(model, X_sea, y_sea, input_size, num_of_units, num_of_epochs, num_of_tasks, batch_size)


    input_size = 7
    num_of_units = 10
    num_of_classes = 2
    model = nn(input_size, num_of_units, num_of_classes)

    stagger_path = 'data/Stagger/stagger_w_50_n_0.1_101.arff'
    X_stagger, y_stagger = preprocess_Stagger(stagger_path)

    # X_stagger = X_stagger[:33333, :]
    # y_stagger = y_stagger[:33333, :]
    # model.fit(X_stagger, y_stagger, epochs=num_of_epochs, batch_size=batch_size, verbose=2, validation_split=0.1)

    training(model, X_stagger, y_stagger, input_size, num_of_units, num_of_epochs, num_of_tasks, batch_size)



