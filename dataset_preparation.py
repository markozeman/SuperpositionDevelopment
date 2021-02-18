from math import floor
# from keras.datasets import cifar100, mnist
# from keras.utils import to_categorical
from tensorflow.keras.datasets import cifar100, mnist
from tensorflow.keras.utils import to_categorical
import numpy as np


def get_CIFAR_100():
    """
    Dataset of 50.000 32x32 color training images, labeled over 100 categories, and 10,000 test images.

    :return: tuple of X_train, y_train, X_test, y_test
    """
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
    return X_train, y_train, X_test, y_test


def get_MNIST():
    """
    Dataset of 60.000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.

    :return: tuple of X_train, y_train, X_test, y_test
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, y_train, X_test, y_test


def disjoint_datasets(X, y):
    """
    Separate bigger dataset to 10 smaller datasets.

    :param X: model input data
    :param y: model output data / label
    :return: 10 disjoint datasets
    """
    sets = [([], []) for _ in range(10)]
    for image, label in zip(X, y):
        index = int(floor(label[0] / 10))
        sets[index][0].append(image)
        sets[index][1].append(to_categorical(label[0] % 10, 10))    # change the last number here for using superfluous neurons
    return sets


def make_disjoint_datasets(input_size, dataset_fun=get_CIFAR_100):
    """
    Make 10 disjoint datasets of the same size from CIFAR-100 or other 'dataset_fun' dataset.

    :param input_size: image input shape
    :param dataset_fun: function that returns specific dataset (default is CIFAR-100 dataset)
    :return: list of 10 disjoint datasets with corresponding train and test set
             [(X_train, y_train, X_test, y_test), (X_train, y_train, X_test, y_test), ...]
    """
    X_train, y_train, X_test, y_test = dataset_fun()
    train_sets = disjoint_datasets(X_train, y_train)
    test_sets = disjoint_datasets(X_test, y_test)
    return list(map(lambda x: (*x[0], *x[1]), zip(train_sets, test_sets)))


def get_dataset(dataset, nn_cnn, input_size, num_of_classes):
    """
    Prepare dataset for input to NN or CNN.

    :param dataset: string: 'mnist' or 'cifar'
    :param nn_cnn: string: 'nn' or 'cnn'
    :param input_size: image input size in pixels
    :param num_of_classes: number of output classes/labels
    :return: (X_train, y_train, X_test, y_test) of MNIST or 10 disjoint sets of CIFAR-100
    """
    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = get_MNIST()

        y_train = to_categorical(y_train, num_classes=num_of_classes)  # one-hot encode
        y_test = to_categorical(y_test, num_classes=num_of_classes)  # one-hot encode

        # normalize input images to have values between 0 and 1
        X_train = X_train.astype(dtype=np.float64)
        X_test = X_test.astype(dtype=np.float64)
        X_train /= 255
        X_test /= 255

        if nn_cnn == 'cnn':
            # reshape to the right dimensions for CNN
            X_train = X_train.reshape(X_train.shape[0], *input_size, 1)
            X_test = X_test.reshape(X_test.shape[0], *input_size, 1)

        return X_train, y_train, X_test, y_test

    elif dataset == 'cifar':
        disjoint_sets = make_disjoint_datasets(input_size)
        for i, dis_set in enumerate(disjoint_sets):
            X_train, y_train, X_test, y_test = dis_set

            # normalize input images to have values between 0 and 1
            X_train = np.array(X_train).astype(dtype=np.float64)
            X_test = np.array(X_test).astype(dtype=np.float64)
            X_train /= 255
            X_test /= 255

            if nn_cnn == 'cnn':
                # reshape to the right dimensions for CNN
                X_train = X_train.reshape(X_train.shape[0], *input_size)
                X_test = X_test.reshape(X_test.shape[0], *input_size)

            y_train = np.array(y_train)
            y_test = np.array(y_test)

            disjoint_sets[i] = (X_train, y_train, X_test, y_test)

        return disjoint_sets


if __name__ == '__main__':
    d = get_dataset('mnist', 'cnn', (28, 28), 10)

