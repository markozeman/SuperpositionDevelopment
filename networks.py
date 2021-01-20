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


class CustomContextLayer(Layer):
    def __init__(self, output_dimension, activation, **kwargs):
        super(CustomContextLayer, self).__init__(**kwargs)
        self.output_dimension = output_dimension
        self.activation = get_keras_activation(activation)

    def build(self, input_shape):
        print('is: ', input_shape)

        self.W = self.add_weight(
            shape=(input_shape[-1], self.output_dimension),
            # initializer="random_normal",
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            shape=(self.output_dimension,),
            # initializer="random_normal",
            initializer="zeros",
            trainable=True
        )

        super(CustomContextLayer, self).build(input_shape)

    def call(self, inputs):
        o = tensorflow.matmul(inputs, self.W) + self.b
        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dimension)


