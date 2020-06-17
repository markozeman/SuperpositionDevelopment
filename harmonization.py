from keras import Sequential, Model
from keras.engine.saving import load_model
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.random_projection import GaussianRandomProjection
import pickle
import tensorflow as tf
import numpy as np
from dataset_preparation import get_dataset
from help_functions import zero_out_vector, get_context_matrices, random_binary_array
from plots import plot_weights_histogram, plot_general
from superposition import superposition_training_cifar


def simple_nn(input_size, num_of_classes):
    """
    Create simple NN model with one hidden layer.

    :param input_size: vector input size
    :param num_of_classes: number of different classes/output labels
    :return: Keras model instance
    """
    model = Sequential()
    model.add(Dense(1000, activation='relu', input_shape=(input_size, )))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def get_feature_vector_representation(datasets, proportion_0=0.0):
    """
    Load trained CNN model for 'datasets' and get representation vectors for all images
    after convolutional and pooling layers. Train and test labels do not change.

    :param datasets: list of disjoint datasets with corresponding train and test set
    :param proportion_0: share of zeros we want in vector to make it more sparse, default=0 which does not change original vector
    :return: 'datasets' images represented as feature vectors
             [(X_train_vectors, y_train, X_test_vectors, y_test), (X_train_vectors, y_train, X_test_vectors, y_test), ...]
    """
    i = 0
    vectors = []
    rnd = np.random.RandomState(42)

    model = load_model('saved_data/CNN_model_Conv_part.h5')
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    conv_contexts = pickle.load(open('saved_data/conv_layers_contexts.pickle', 'rb'))

    # save current Conv weights
    curr_w_matrices = []
    curr_bias_vectors = []
    for layer_index, layer in enumerate(model.layers):
        if 0 < layer_index < 3:  # conv layer
            curr_w_matrices.append(layer.get_weights()[0])
            curr_bias_vectors.append(layer.get_weights()[1])

    for X_train, y_train, X_test, y_test in datasets:
        print('i: ', i)

        # unfold weights of 'model' with 'conv_contexts'
        for layer_index, layer in enumerate(model.layers):
            if 0 < layer_index < 3:   # conv layer
                context_vector = conv_contexts[9][layer_index - 1]  # [9] because it is the index of the last context
                for task_i in range(9, i, -1):
                    context_vector = np.multiply(context_vector, conv_contexts[task_i][layer_index - 1])

                new_w = np.reshape(np.multiply(curr_w_matrices[layer_index - 1].flatten(), context_vector), curr_w_matrices[layer_index - 1].shape)
                layer.set_weights([new_w, curr_bias_vectors[layer_index - 1]])

        X_train_vectors = model.predict(X_train)
        X_test_vectors = model.predict(X_test)

        for index in range(X_train_vectors.shape[0]):
            X_train_vectors[index] = zero_out_vector(X_train_vectors[index], proportion_0)   # only zero out elements

        for index in range(X_test_vectors.shape[0]):
            X_test_vectors[index] = zero_out_vector(X_test_vectors[index], proportion_0)  # only zero out elements

        # plot_weights_histogram(X_train_vectors[0], 30)  # to test new weights distribution

        # perform random projection on feature representations to reduce dimensionality
        transformer = GaussianRandomProjection(n_components=1000, random_state=rnd)
        XX = np.vstack((X_train_vectors, X_test_vectors))
        XX = transformer.fit_transform(XX)
        X_train_vectors = XX[:5000, :]
        X_test_vectors = XX[5000:, :]

        vectors.append((X_train_vectors, y_train, X_test_vectors, y_test))

        i += 1

    return vectors


if __name__ == '__main__':
    # to avoid cuDNN error (https://github.com/tensorflow/tensorflow/issues/24496)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    num_of_classes = 10
    num_of_epochs = 10
    batch_size = 50

    dataset = 'cifar'
    nn_cnn = 'cnn'
    input_size = (32, 32, 3)

    d = get_dataset(dataset, nn_cnn, input_size, num_of_classes)

    datasets = get_feature_vector_representation(d, proportion_0=0.92028)   # from 12.544 to 1.000 units

    nn_cnn = 'nn'
    input_size = 1000
    num_of_units = 1000
    num_of_tasks = 10
    model = simple_nn(input_size, num_of_classes)

    # get context_matrices for simple NN model with one hidden layer.
    context_matrices = [[random_binary_array(input_size), random_binary_array(1000)] for i in range(num_of_tasks)]

    acc_superposition = superposition_training_cifar(model, datasets, num_of_epochs, num_of_tasks, context_matrices, nn_cnn, batch_size)

    plot_general(acc_superposition, [], ['Superposition with harmonization'],
                 '', 'Epoch', 'Accuracy (%)', [10], 0, 60)



