import tensorflow as tf
from callbacks import *
from help_functions import *
from plots import *
from networks import *
from dataset_preparation import get_dataset


def train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size=32, validation_share=0.0,
                mode='normal', context_matrices=None, task_index=None):
    """
    Train and evaluate Keras model.

    :param model: Keras model instance
    :param X_train: train input data
    :param y_train: train output labels
    :param X_test: test input data
    :param y_test: test output labels
    :param num_of_epochs: number of epochs to train the model
    :param batch_size: batch size - number of samples per gradient update (default = 32)
    :param validation_share: share of examples to be used for validation (default = 0)
    :param mode: string for learning mode, important for callbacks - possible values: 'normal', 'superposition'
    :param context_matrices: multidimensional numpy array with random context (binary superposition), only used when mode = 'superposition'
    :param task_index: index of current task, only used when mode = 'superposition'
    :return: History object and 2 lists of test accuracies for every training epoch (normal and superposition)
    """
    test_callback = TestPerformanceCallback(X_test, y_test, model)
    test_superposition_callback = TestSuperpositionPerformanceCallback(X_test, y_test, context_matrices, model, task_index)
    lr_callback = LearningRateScheduler(lr_scheduler)

    callbacks = [lr_callback]
    if mode == 'normal':
        callbacks.append(test_callback)
    elif mode == 'superposition':
        callbacks.append(test_superposition_callback)

    history = model.fit(X_train, y_train, epochs=num_of_epochs, batch_size=batch_size, verbose=2,
                        validation_split=validation_share, callbacks=callbacks)
    return history, test_callback.accuracies, test_superposition_callback.accuracies


def normal_training_mnist(model, X_train, y_train, X_test, y_test, num_of_epochs, num_of_tasks, batch_size=32):
    """
    Train model for 'num_of_tasks' tasks, each task is a different permutation of input images.
    Check how accuracy for original images is changing through tasks using normal training.

    :param model: Keras model instance
    :param X_train: train input data
    :param y_train: train output labels
    :param X_test: test input data
    :param y_test: test output labels
    :param num_of_epochs: number of epochs to train the model
    :param num_of_tasks: number of different tasks (permutations of original images)
    :param batch_size: batch size - number of samples per gradient update (default = 32)
    :return: list of test accuracies for 10 epochs for each task
    """
    original_accuracies = []

    # first training task - original MNIST images
    history, accuracies, _ = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size, validation_share=0.1)
    original_accuracies.extend(accuracies)
    print_validation_acc(history, 0)

    # other training tasks - permuted MNIST data
    for i in range(num_of_tasks - 1):
        print("\n\n Task: %d \n" % (i + 1))

        permuted_X_train = permute_images(X_train)
        history, accuracies, _ = train_model(model, permuted_X_train, y_train, X_test, y_test, num_of_epochs, batch_size, validation_share=0.1)

        original_accuracies.extend(accuracies)
        print_validation_acc(history, i + 1)

    return original_accuracies


def superposition_training_mnist(model, X_train, y_train, X_test, y_test, num_of_epochs, num_of_tasks, context_matrices, batch_size=32):
    """
    Train model for 'num_of_tasks' tasks, each task is a different permutation of input images.
    Check how accuracy for original images is changing through tasks using superposition training.

    :param model: Keras model instance
    :param X_train: train input data
    :param y_train: train output labels
    :param X_test: test input data
    :param y_test: test output labels
    :param num_of_epochs: number of epochs to train the model
    :param num_of_tasks: number of different tasks (permutations of original images)
    :param context_matrices: multidimensional numpy array with random context (binary superposition)
    :param batch_size: batch size - number of samples per gradient update (default = 32)
    :return: list of test accuracies for 10 epochs for each task
    """
    original_accuracies = []

    # context_multiplication(model, context_matrices, 0)

    # first training task - original MNIST images
    history, _, accuracies = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, batch_size, validation_share=0.1,
                                         mode='superposition', context_matrices=context_matrices, task_index=0)
    original_accuracies.extend(accuracies)
    print_validation_acc(history, 0)

    # other training tasks - permuted MNIST data
    for i in range(num_of_tasks - 1):
        print("\n\n Task: %d \n" % (i + 1))

        # multiply current weights with context matrices for each layer (without changing weights from bias node)
        context_multiplication(model, context_matrices, i + 1)

        permuted_X_train = permute_images(X_train)
        history, _, accuracies = train_model(model, permuted_X_train, y_train, X_test, y_test, num_of_epochs, batch_size, validation_share=0.1,
                                             mode='superposition', context_matrices=context_matrices, task_index=i + 1)

        original_accuracies.extend(accuracies)
        print_validation_acc(history, i + 1)

    return original_accuracies


if __name__ == '__main__':
    # to avoid cuDNN error (https://github.com/tensorflow/tensorflow/issues/24496)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    dataset = 'mnist'   # 'mnist' or 'cifar'
    nn_cnn = 'nn'      # 'nn' or 'cnn'
    input_size = (28, 28) if dataset == 'mnist' else (32, 32, 3)    # input sizes for MNIST and CIFAR images
    num_of_units = 1000
    num_of_classes = 10

    num_of_tasks = 5
    num_of_epochs = 10
    batch_size = 600

    train_normal = True
    train_superposition = True

    if train_normal:
        d = get_dataset(dataset, nn_cnn, input_size, num_of_classes)
        X_train, y_train, X_test, y_test = d

        if nn_cnn == 'nn':
            model = nn(input_size, num_of_units, num_of_classes)
        elif nn_cnn == 'cnn':
            model = cnn(input_size, num_of_classes)
        else:
            raise ValueError("nn_cnn variable must have value 'nn' or 'cnn'")

        acc_normal = normal_training_mnist(model, X_train, y_train, X_test, y_test, num_of_epochs, num_of_tasks, batch_size)
        print(acc_normal)

    if train_superposition:
        d = get_dataset(dataset, nn_cnn, input_size, num_of_classes)
        X_train, y_train, X_test, y_test = d

        if nn_cnn == 'nn':
            model = nn(input_size, num_of_units, num_of_classes)
            context_matrices = get_context_matrices(input_size, num_of_units, num_of_tasks)
        elif nn_cnn == 'cnn':
            model = cnn(input_size, num_of_classes)
            context_matrices = get_context_matrices_CNN(model, num_of_tasks)
        else:
            raise ValueError("nn_cnn variable must have value 'nn' or 'cnn'")

        acc_superposition = superposition_training_mnist(model, X_train, y_train, X_test, y_test, num_of_epochs,
                                                         num_of_tasks, context_matrices, batch_size)
        print(acc_superposition)

    plot_accuracies_over_time(acc_normal, acc_superposition)


