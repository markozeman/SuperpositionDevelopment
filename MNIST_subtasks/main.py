import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
from dataset_preparation import get_dataset
from superposition import normal_training_mnist, nn, plot_confusion_matrix
from sklearn.metrics import confusion_matrix


def extract_digits(X, y, digits):
    """
    Extract the given digits from X and y and return only those digits.

    :param X: list of images
    :param y: list of class labels
    :param digits: list of digits to extract
    :return: extracted X, extracted y
    """
    one_hot_digits = [list(to_categorical(digit, 10)) for digit in digits]
    X_extracted = []
    y_extracted = []
    for i, label in enumerate(y):
        if np.any([np.array_equal(digit, label) for digit in one_hot_digits]):
            X_extracted.append(X[i])
            y_extracted.append(label)
    return np.array(X_extracted), np.array(y_extracted)


if __name__ == '__main__':
    # to avoid cuDNN error (https://github.com/tensorflow/tensorflow/issues/24496)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    dataset = 'mnist'
    nn_cnn = 'nn'
    input_size = (28, 28)
    num_of_units = 1000
    num_of_classes = 10

    num_of_tasks = 1
    num_of_epochs = 10
    batch_size = 600

    X_train, y_train, X_test, y_test = get_dataset(dataset, nn_cnn, input_size, num_of_classes)
    model = nn(input_size, num_of_units, num_of_classes)

    # extract only some digits for training
    X_train, y_train = extract_digits(X_train, y_train, [8,9])
    X_test, y_test = extract_digits(X_test, y_test, [8,9])

    acc_normal = normal_training_mnist(model, X_train, y_train, X_test, y_test, num_of_epochs, num_of_tasks, nn_cnn, batch_size)

    # prepare confusion matrix
    y_predicted = model.predict_classes(X_test)
    y_true = np.argmax(y_test, axis=1)      # reverse one-hot encoding
    conf_mat = confusion_matrix(y_true, y_predicted)
    plot_confusion_matrix(conf_mat)




