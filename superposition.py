import tensorflow as tf
from callbacks import *
from help_functions import *
from networks import *
from dataset_preparation import get_dataset


if __name__ == '__main__':
    # to avoid cuDNN error (https://github.com/tensorflow/tensorflow/issues/24496)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    dataset = 'cifar'   # 'mnist' or 'cifar'
    nn_cnn = 'cnn'      # 'nn' or 'cnn'
    input_size = (28, 28) if dataset == 'mnist' else (32, 32, 3)    # input sizes for MNIST and CIFAR images
    num_of_units = 1000
    num_of_classes = 10

    num_of_tasks = 3
    num_of_epochs = 10
    batch_size = 600

    d = get_dataset(dataset, nn_cnn, input_size, num_of_classes)
    X_train, y_train, X_test, y_test = d[0]

    if nn_cnn == 'nn':
        model = nn(input_size, num_of_units, num_of_classes)
    elif nn_cnn == 'cnn':
        model = cnn(input_size, num_of_classes)

    history = model.fit(X_train, y_train, epochs=num_of_epochs, batch_size=batch_size, verbose=2, validation_split=0.1)


