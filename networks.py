import sys

import numpy as np
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Layer
from keras.optimizers import Adam
from keras.activations import get as get_keras_activation

## to enable always the same initialization of the networks
from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)
import tensorflow
tensorflow.random.set_seed(2)


def nn(input_size, num_of_units, num_of_classes):
    """
    Create simple NN model with two hidden layers, each has 'num_of_units' neurons.

    :param input_size: image input size in pixels
    :param num_of_units: number of neurons in each hidden layer
    :param num_of_classes: number of different classes/output labels
    :return: Keras model instance
    """
    model = Sequential()
    model.add(Flatten(input_shape=input_size))
    model.add(Dense(num_of_units, activation='relu'))
    model.add(Dense(num_of_units, activation='relu'))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def cnn(input_size, num_of_classes):
    """
    Create simple CNN model.

    :param input_size: image input size in pixels
    :param num_of_classes: number of different classes/output labels
    :return: Keras model instance
    """
    if len(input_size) == 2:    # grayscale image (mnist)
        input_size = (*input_size, 1)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_size))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


@tensorflow.keras.utils.register_keras_serializable()
class CustomContextLayer(Layer):
    """
    Custom Keras Layer for training the contexts.
    """
    def __init__(self, output_dimension, activation, **kwargs):
        """
        Initialize custom layer instance.

        :param output_dimension: output size of this custom layer
        :param activation: activation function that is used on the output of this custom layer
        :param kwargs: possible additional arguments
        """
        super(CustomContextLayer, self).__init__(**kwargs)
        self.output_dimension = output_dimension
        self.activation = get_keras_activation(activation)

    def build(self, input_shape):
        """
        Definition of the layer's weights/parameters.

        :param input_shape: input dimension of this custom layer
        :return: None
        """
        self.W = self.add_weight(
            shape=(input_shape[-1], ),
            initializer="glorot_uniform",
            # initializer="random_normal",
            # initializer="zeros",
            # regularizer=custom_regularizer_to_zero,
            # constraint=tensorflow.keras.constraints.MinMaxNorm(min_value=-1.01, max_value=1.01, rate=0.00001, axis=0),
            trainable=True
        )

        # print('build, input_shape: ', input_shape)
        # print('build, self.W.shape:', self.W.shape)
        # print(self.W)

        super(CustomContextLayer, self).build(input_shape)

    def call(self, inputs):
        """
        Behaviour of this layer - feed-forward pass.

        :param inputs: the input tensor from the previous layer
        :return: output of this custom layer after activation function
        """
        # o = tensorflow.math.multiply(inputs, self.W)
        o = tensorflow.math.multiply(inputs, sigmoid_from_minus1_to_1(self.W))
        # o = tensorflow.math.multiply(inputs, factor_sigmoid_from_minus1_to_1(self.W))

        return self.activation(o)

    def compute_output_shape(self, input_shape):
        """
        Specify the change in shape of the input when it passes through the layer.

        :param input_shape: input dimension of this custom layer
        :return: output shape of this custom layer
        """
        return (input_shape[0], self.output_dimension)

    def get_config(self):
        """
        Method for serialization of the custom layer.

        :return: dict with needed keys and values
        """
        config = {
            'output_dimension': self.output_dimension,
            'activation': self.activation
        }
        return config


def sigmoid_from_minus1_to_1(x):
    """
    Corrected sigmoid activation function that outputs value between -1 and 1.

    :param x: input into the activation function
    :return: output value
    """
    from keras import backend as K
    return (K.sigmoid(x) * 2) - 1


def factor_sigmoid_from_minus1_to_1(x):
    """
    Corrected sigmoid activation function that outputs value between -1 and 1.
    Factor is used for the slope of the function around 0.

    :param x: input into the activation function
    :return: output value
    """
    # import matplotlib.pyplot as plt
    # plt.plot(np.linspace(-10, 10, 2001), list(map(factor_sigmoid_from_minus1_to_1, np.linspace(-10, 10, 2001))))
    # plt.show()
    factor = 10
    return ((1 / (1 + tensorflow.math.exp(factor * -x))) * 2) - 1


def custom_regularizer_to_ones(x):
    """
    Regularizer for custom layers that enforces tensor values to go towards -1 or 1.

    :param x: tensor of weights, which we want to regularize
    :return: value of the loss for the custom layer
    """
    return 0.000005 * tensorflow.reduce_sum(tensorflow.abs((tensorflow.abs(x)) - 1))   # / x.get_shape()[0]


def custom_regularizer_to_zero(x):
    """
    Regularizer for custom layers that enforces tensor values to go towards 0 (the same as L2 regularization).

    :param x: tensor of weights, which we want to regularize
    :return: value of the loss for the custom layer
    """
    return 0.00001 * tensorflow.reduce_sum(tensorflow.square(sigmoid_from_minus1_to_1(x)))



