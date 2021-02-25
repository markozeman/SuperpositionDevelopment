import tensorflow as tf
import random
import pickle
# from keras.callbacks import LearningRateScheduler, LambdaCallback
# from keras.engine.saving import model_from_json
# from keras.layers import BatchNormalization
# from keras.models import load_model, clone_model
# from keras.optimizers import SGD
# from keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

from callbacks import *
from help_functions import *
from plots import *
from networks import *
from dataset_preparation import get_dataset
import multiprocessing


def train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, nn_cnn, batch_size=32, validation_share=0.0,
                mode='normal', context_matrices=None, task_index=None):
    """
    Train and evaluate Keras model.

    :param model: Keras model instance
    :param X_train: train input data
    :param y_train: train output labels
    :param X_test: test input data
    :param y_test: test output labels
    :param num_of_epochs: number of epochs to train the model
    :param nn_cnn: usage of (convolutional) neural network (possible values: 'nn' or 'cnn')
    :param batch_size: batch size - number of samples per gradient update (default = 32)
    :param validation_share: share of examples to be used for validation (default = 0)
    :param mode: string for learning mode, important for callbacks - possible values: 'normal', 'superposition'
    :param context_matrices: multidimensional numpy array with random context (binary superposition), only used when mode = 'superposition'
    :param task_index: index of current task, only used when mode = 'superposition'
    :return: History object and 2 lists of test accuracies for every training epoch (normal and superposition)
    """
    lr_callback = LearningRateScheduler(lr_scheduler)
    test_callback = TestPerformanceCallback(X_test, y_test, model)
    if nn_cnn == 'nn':
        test_superposition_callback = TestSuperpositionPerformanceCallback(X_test, y_test, context_matrices, model, task_index)
    elif nn_cnn == 'cnn':
        test_superposition_callback = TestSuperpositionPerformanceCallback_CNN(X_test, y_test, context_matrices, model, task_index)

    # callbacks = [lr_callback]
    callbacks = []      # todo
    if mode == 'normal':
        callbacks.append(test_callback)
    elif mode == 'superposition':
        callbacks.append(test_superposition_callback)

    history = model.fit(X_train, y_train, epochs=num_of_epochs, batch_size=batch_size, verbose=2,
                        validation_split=validation_share, callbacks=callbacks)

    # global lr_over_time
    # plot_many_lines([lr_over_time], ['LR'], 'Learning rate through training epochs', 'epoch', 'learning rate')

    # print('LRs:', test_superposition_callback.LR)
    # plot_many_lines([test_superposition_callback.LR], ['LR'], 'Learning rate through training iterations', 'iteration', 'learning rate')

    return history, test_callback.accuracies, test_superposition_callback.accuracies


def normal_training_mnist(model, X_train, y_train, X_test, y_test, num_of_epochs, num_of_tasks, nn_cnn, batch_size=32):
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
    :param nn_cnn: usage of (convolutional) neural network (possible values: 'nn' or 'cnn')
    :param batch_size: batch size - number of samples per gradient update (default = 32)
    :return: list of test accuracies for num_of_epochs epochs for each task
    """
    original_accuracies = []

    # L_before = [np.sign(model.layers[1].get_weights()[0]), np.sign(model.layers[2].get_weights()[0]), np.sign(model.layers[3].get_weights()[0])]

    # first training task - original MNIST images
    history, accuracies, _ = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, nn_cnn, batch_size, validation_share=0.1)
    original_accuracies.extend(accuracies)
    print_validation_acc(history, 0)

    # L_after = [np.sign(model.layers[1].get_weights()[0]), np.sign(model.layers[2].get_weights()[0]), np.sign(model.layers[3].get_weights()[0])]
    # compare_weights_signs(L_before, L_after)

    # other training tasks - permuted MNIST data
    for i in range(num_of_tasks - 1):
        print("\n\n Task: %d \n" % (i + 1))

        # L_before = [np.sign(model.layers[1].get_weights()[0]), np.sign(model.layers[2].get_weights()[0]), np.sign(model.layers[3].get_weights()[0])]

        permuted_X_train = permute_images(X_train, i)
        history, accuracies, _ = train_model(model, permuted_X_train, y_train, X_test, y_test, num_of_epochs, nn_cnn, batch_size, validation_share=0.1)

        # L_after = [np.sign(model.layers[1].get_weights()[0]), np.sign(model.layers[2].get_weights()[0]), np.sign(model.layers[3].get_weights()[0])]
        # compare_weights_signs(L_before, L_after)

        original_accuracies.extend(accuracies)
        print_validation_acc(history, i + 1)

    return original_accuracies


def superposition_training_mnist(model, X_train, y_train, X_test, y_test, num_of_epochs, num_of_tasks, context_matrices, nn_cnn, batch_size=32):
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
    :param nn_cnn: usage of (convolutional) neural network (possible values: 'nn' or 'cnn')
    :param batch_size: batch size - number of samples per gradient update (default = 32)
    :return: list of test accuracies for num_of_epochs epochs for each task
    """
    num_of_units = len(context_matrices[0][1])
    original_accuracies = []
    show_W_heatmaps = False

    # context_multiplication(model, context_matrices, 0)

    if nn_cnn == 'nn':
        W_before = model.layers[3].get_weights()[0]

    L_before = [np.sign(model.layers[1].get_weights()[0]), np.sign(model.layers[2].get_weights()[0]), np.sign(model.layers[3].get_weights()[0])]

    # first training task - original MNIST images
    history, _, accuracies = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, nn_cnn, batch_size, validation_share=0.1,
                                         mode='superposition', context_matrices=context_matrices, task_index=0)
    original_accuracies.extend(accuracies)
    print_validation_acc(history, 0)

    L_after = [np.sign(model.layers[1].get_weights()[0]), np.sign(model.layers[2].get_weights()[0]), np.sign(model.layers[3].get_weights()[0])]
    compare_weights_signs(L_before, L_after)

    # model.save("my_tmp_model_3.h5")

    # plot_weights_histogram(model.layers[3].get_weights()[0].flatten(), 30)

    if nn_cnn == 'nn':
        W_after = model.layers[3].get_weights()[0]
        W_diff = np.absolute(W_before - W_after)    # absolute difference of weight matrices before and after training
        if show_W_heatmaps:
            weights_heatmaps([W_before, W_after, W_diff], ['before 1st training', 'after 1st training', 'diff'], 0)

    # other training tasks - permuted MNIST data
    for i in range(num_of_tasks - 1):
        print("\n\n Task: %d \n" % (i + 1))

        if nn_cnn == 'nn':
            W_before = model.layers[3].get_weights()[0]

        ### Find the best context for the current task and use it instead of a random context
        learn_context = False
        if learn_context:
            num_of_epochs_context = 10

            # deep copy model into model_context
            model_context = clone_model(model)
            model_context.build((None, 784))
            model_context.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
            model_context.set_weights(model.get_weights())

            # model_context = load_model("my_tmp_model_3.h5")

            # insert custom layers
            model_context = insert_intermediate_layer_in_keras(model_context, 1, CustomContextLayer(784, activation='linear'))
            model_context = insert_intermediate_layer_in_keras(model_context, 4, CustomContextLayer(num_of_units, activation='linear'))
            model_context = insert_intermediate_layer_in_keras(model_context, 6, CustomContextLayer(num_of_units, activation='linear'))

            # Dense layers not trainable
            model_context.layers[3].trainable = False
            model_context.layers[5].trainable = False
            model_context.layers[7].trainable = False

            model_context.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
            model_context.summary()

            # print(model_context.layers[6].get_weights()[0])

            callback_discrete_acc = PrintDiscreteAccuracy(permute_images(X_test, i), y_test, model_context, context_matrices)
            permuted_images = permute_images(X_train, i)
            model_context.fit(permuted_images, y_train, epochs=num_of_epochs_context, verbose=2, validation_split=0.1,
                              callbacks=[callback_discrete_acc])

            print_number_of_changed_context_signs(callback_discrete_acc.starting_context_values, callback_discrete_acc.last_context_values)

            # print(model_context.layers[6].get_weights()[0])

            # override random context with learned context
            context_matrices[i + 1] = callback_discrete_acc.last_context_values
            # print('CM:', context_matrices[i + 1][2])

        # model = load_model("my_tmp_model_3.h5")

        # print('sums:', sum(context_matrices[1][0]), sum(context_matrices[1][1]), sum(context_matrices[1][2]))

        # multiply current weights with context matrices for each layer (without changing weights from bias node)
        if nn_cnn == 'nn':
            context_multiplication(model, context_matrices, i + 1)
        elif nn_cnn == 'cnn':
            context_multiplication_CNN(model, context_matrices, i + 1)

        # # to enable dynamic contexts - update them while learning, when they are not needed for weight change anymore
        # # in the callback you can only multiply with the specific dynamic context vector then to get the right weights
        # if i > 0:
        #     for task_index in range(1, i + 1):
        #         for layer_index in range(len(context_matrices[0])):
        #             context_matrices[task_index][layer_index] = np.multiply(context_matrices[task_index][layer_index],
        #                                                                     context_matrices[i + 1][layer_index])

        if nn_cnn == 'nn':
            W_after = model.layers[3].get_weights()[0]
            W_diff = np.absolute(W_before - W_after)  # difference of weight matrices before and after context multiplication

        '''
        # 1,5h of training - best accuracy: 38.9
        round_index = 0
        while True:
            if round_index != 0:
                # perform changes of signs that proved the best
                for l_0_index in best_l_0:
                    context_matrices[i + 11][0][l_0_index] = -1 * context_matrices[i + 11][0][l_0_index]
                context_matrices[i + 11][1][best_l_1] = -1 * context_matrices[i + 11][1][best_l_1]
                context_matrices[i + 11][2][best_l_2] = -1 * context_matrices[i + 11][2][best_l_2]
    
            best_index, best_value = 0, 0
            best_l_0, best_l_1, best_l_2 = None, None, None
            for number_of_options in range(50):
                model = load_model("model_after1task_30.h5")
    
                layer_0 = [random.randint(0, 783) for _ in range(10)]
                layer_1 = random.randint(0, 29)
                layer_2 = random.randint(0, 29)
    
                # apply changes to contexts
                for l_0_index in layer_0:
                    context_matrices[i + 11][0][l_0_index] = -1 * context_matrices[i + 11][0][l_0_index]
                context_matrices[i + 11][1][layer_1] = -1 * context_matrices[i + 11][1][layer_1]
                context_matrices[i + 11][2][layer_2] = -1 * context_matrices[i + 11][2][layer_2]
    
                context_multiplication(model, context_matrices, i + 11, None)
    
                results = model.evaluate(permute_images(X_test, i), y_test, verbose=0)
                print('index:' , number_of_options, 'test acc: ', round(results[1] * 100, 2))
    
                # save result if it's the best until now
                if round(results[1] * 100, 2) > best_value:
                    best_value = round(results[1] * 100, 2)
                    best_index = number_of_options
                    best_l_0, best_l_1, best_l_2 = layer_0, layer_1, layer_2
    
                # reverse context
                for l_0_index in layer_0:
                    context_matrices[i + 11][0][l_0_index] = -1 * context_matrices[i + 11][0][l_0_index]
                context_matrices[i + 11][1][layer_1] = -1 * context_matrices[i + 11][1][layer_1]
                context_matrices[i + 11][2][layer_2] = -1 * context_matrices[i + 11][2][layer_2]
    
            print('BEST: ', best_index, best_value)
            print(best_l_0, best_l_1, best_l_2, '\n')
            round_index += 1
        '''

        '''
        permuted_images = permute_images(X_test, i)
        round_index = 0
        num_of_units = len(context_matrices[0][1])
        model = load_model("temp_model_%s.h5" % str(num_of_units))
        context_matrices[i + 11] = np.load('best_context_vectors_%s.npy' % str(num_of_units), allow_pickle=True)
        best_acc = np.load('best_accuracy_%s.npy' % str(num_of_units))
        print(best_acc)
        while True:
            # all_changing_neurons = 784 + 2 * num_of_units
            # lyr = np.random.choice(np.arange(0, 3),
            #                        p=[784 / all_changing_neurons, num_of_units / all_changing_neurons, num_of_units / all_changing_neurons])
    
            lyr = random.randint(0, 2)  # evenly distributed 0, 1 or 2
            nrn = random.randint(0, num_of_units - 1) if lyr > 0 else random.randint(0, 783)
    
            # apply change to contexts
            context_matrices[i + 11][lyr][nrn] = -1 * context_matrices[i + 11][lyr][nrn]
    
            context_multiplication(model, context_matrices, i + 11)
    
            results = model.evaluate(permuted_images, y_test, verbose=0)
            acc = round(results[1] * 100, 2)
            print('round index:', round_index, 'test acc: ', acc)
    
            # revert model weights back
            context_multiplication(model, context_matrices, i + 11)
    
            # save result if it's the best until now
            if acc > best_acc:
                best_acc = acc
                print('BEST acc: ', best_acc)
                print('layer', lyr)
                np.save('best_accuracy_%s.npy' % str(num_of_units), best_acc)
                np.save('best_context_vectors_%s.npy' % str(num_of_units), context_matrices[i + 11])
            else:  # reverse context
                context_matrices[i + 11][lyr][nrn] = -1 * context_matrices[i + 11][lyr][nrn]
    
            round_index += 1
        
        
        return
        '''

        L_before = [np.sign(model.layers[1].get_weights()[0]), np.sign(model.layers[2].get_weights()[0]), np.sign(model.layers[3].get_weights()[0])]

        # weights_heatmaps([model.layers[3].get_weights()[0]], ['pred'], 1)

        permuted_X_train = permute_images(X_train, i)
        history, _, accuracies = train_model(model, permuted_X_train, y_train, X_test, y_test, num_of_epochs, nn_cnn, batch_size, validation_share=0.1,
                                             mode='superposition', context_matrices=context_matrices, task_index=i + 1)

        # weights_heatmaps([model.layers[3].get_weights()[0]], ['po'], 1)

        L_after = [np.sign(model.layers[1].get_weights()[0]), np.sign(model.layers[2].get_weights()[0]), np.sign(model.layers[3].get_weights()[0])]
        compare_weights_signs(L_before, L_after)

        if nn_cnn == 'nn':
            W_after_training = model.layers[3].get_weights()[0]
            W_diff_training = np.absolute(W_after_training - W_after)  # difference of weight matrices before and after training

            if show_W_heatmaps:
                # print('context: ', context_matrices[1][2])
                # W_T0_unfolded = np.diag(context_matrices[1][2] * context_matrices[2][2] * context_matrices[3][2] * context_matrices[4][2]) @ W_after_training  # first task, final layer
                # W_T0_init_vs_unfolded = np.absolute(W_T0_unfolded - W_before)

                # # check if all weights' signs are equal before and after training of one task
                # for a, b in zip(W_after.flatten(), W_after_training.flatten()):
                #     if np.sign(a) != np.sign(b):
                #         print('a, b: ', a, b)

                weights_heatmaps([W_before, W_after, W_diff, W_diff_training, W_after_training],
                                 ['before context mul.', 'after context mul.', 'diff context', 'diff training', 'after training'], i + 1)

        original_accuracies.extend(accuracies)
        print_validation_acc(history, i + 1)

    return original_accuracies


def normal_training_cifar(model, datasets, num_of_epochs, num_of_tasks, nn_cnn, batch_size=32):
    """
    Train model for 'num_of_tasks' tasks, each task is a different disjoint set of CIFAR-100 images.
    Check how accuracy for the first set of images is changing through tasks using normal training.

    :param model: Keras model instance
    :param datasets: list of disjoint datasets with corresponding train and test set
    :param num_of_epochs: number of epochs to train the model
    :param num_of_tasks: number of different tasks
    :param nn_cnn: usage of (convolutional) neural network (possible values: 'nn' or 'cnn')
    :param batch_size: batch size - number of samples per gradient update (default = 32)
    :return: list of test accuracies for num_of_epochs epochs for each task
    """
    original_accuracies = []

    # first training task - 10 classes of CIFAR-100 dataset
    X_train, y_train, X_test, y_test = datasets[0]     # these X_test and y_test are used for testing all tasks
    history, accuracies, _ = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, nn_cnn, batch_size, validation_share=0.1)

    original_accuracies.extend(accuracies)
    print_validation_acc(history, 0)

    # other training tasks
    for i in range(num_of_tasks - 1):
        print("\n\n Task: %d \n" % (i + 1))

        X_train, y_train, _, _ = datasets[i + 1]    # use X_test and y_test from the first task to get its accuracy
        history, accuracies, _ = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, nn_cnn, batch_size, validation_share=0.1)

        original_accuracies.extend(accuracies)
        print_validation_acc(history, i + 1)

    return original_accuracies


def superposition_training_cifar(model, datasets, num_of_epochs, num_of_tasks, context_matrices, nn_cnn, batch_size=32):
    """
    Train model for 'num_of_tasks' tasks, each task is a different disjoint set of CIFAR-100 images.
    Check how accuracy for the first set of images is changing through tasks using superposition training.

    :param model: Keras model instance
    :param datasets: list of disjoint datasets with corresponding train and test set
    :param num_of_epochs: number of epochs to train the model
    :param num_of_tasks: number of different tasks
    :param context_matrices: multidimensional numpy array with random context (binary superposition)
    :param nn_cnn: usage of (convolutional) neural network (possible values: 'nn' or 'cnn')
    :param batch_size: batch size - number of samples per gradient update (default = 32)
    :return: list of test accuracies for num_of_epochs epochs for each task
    """
    original_accuracies = []

    # first training task - 10 classes of CIFAR-100 dataset
    X_train, y_train, X_test, y_test = datasets[0]  # these X_test and y_test are used for testing all tasks
    history, _, accuracies = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, nn_cnn, batch_size, validation_share=0.1,
                                         mode='superposition', context_matrices=context_matrices, task_index=0)

    original_accuracies.extend(accuracies)
    print_validation_acc(history, 0)

    # other training tasks
    for i in range(num_of_tasks - 1):
        print("\n\n i: %d \n" % i)

        # multiply current weights with context matrices for each layer (without changing weights from bias node)
        if nn_cnn == 'nn':
            context_multiplication(model, context_matrices, i + 1)
        elif nn_cnn == 'cnn':
            context_multiplication_CNN(model, context_matrices, i + 1)

        X_train, y_train, _, _ = datasets[i + 1]    # use X_test and y_test from the first task to get its accuracy
        history, _, accuracies = train_model(model, X_train, y_train, X_test, y_test, num_of_epochs, nn_cnn, batch_size, validation_share=0.1,
                                             mode='superposition', context_matrices=context_matrices, task_index=i + 1)

        original_accuracies.extend(accuracies)
        print_validation_acc(history, i + 1)

    return original_accuracies


if __name__ == '__main__':
    # to avoid cuDNN error (https://github.com/tensorflow/tensorflow/issues/24496)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    # to avoid CUBLAS_STATUS_ALLOC_FAILED error (https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    dataset = 'mnist'   # 'mnist' or 'cifar'
    nn_cnn = 'nn'      # 'nn' or 'cnn'
    input_size = (28, 28) if dataset == 'mnist' else (32, 32, 3)    # input sizes for MNIST and CIFAR images
    num_of_units = 30
    num_of_classes = 10     # or number of neurons together with superfluous neurons for 'mnist'
    # (for 'cifar' change function disjoint_datasets in dataset_preparation.py)

    num_of_tasks = 2
    num_of_epochs = 100
    batch_size = 600 if dataset == 'mnist' else 50

    train_normal = False
    train_superposition = True

    if train_normal:
        if nn_cnn == 'nn':
            model = nn(input_size, num_of_units, num_of_classes)
        elif nn_cnn == 'cnn':
            model = cnn(input_size, num_of_classes)
        else:
            raise ValueError("'nn_cnn' variable must have value 'nn' or 'cnn'")

        d = get_dataset(dataset, nn_cnn, input_size, num_of_classes)
        if dataset == 'mnist':
            X_train, y_train, X_test, y_test = d
            acc_normal = normal_training_mnist(model, X_train, y_train, X_test, y_test, num_of_epochs, num_of_tasks, nn_cnn, batch_size)
        elif dataset == 'cifar':
            acc_normal = normal_training_cifar(model, d, num_of_epochs, num_of_tasks, nn_cnn, batch_size)
        else:
            raise ValueError("'dataset' variable must have value 'mnist' or 'cifar'")

    if train_superposition:
        if nn_cnn == 'nn':
            model = nn(input_size, num_of_units, num_of_classes)
            context_matrices = get_context_matrices(input_size, num_of_units, num_of_tasks)
        elif nn_cnn == 'cnn':
            model = cnn(input_size, num_of_classes)
            context_matrices = get_context_matrices_CNN(model, num_of_tasks)
        else:
            raise ValueError("nn_cnn variable must have value 'nn' or 'cnn'")

        # plot_weights_histogram(model.layers[3].get_weights()[0].flatten(), 30)
        # print(context_matrices[1])

        d = get_dataset(dataset, nn_cnn, input_size, num_of_classes)
        if dataset == 'mnist':
            X_train, y_train, X_test, y_test = d
            acc_superposition = superposition_training_mnist(model, X_train, y_train, X_test, y_test, num_of_epochs,
                                                             num_of_tasks, context_matrices, nn_cnn, batch_size)
        elif dataset == 'cifar':
            acc_superposition = superposition_training_cifar(model, d, num_of_epochs, num_of_tasks, context_matrices, nn_cnn, batch_size)
        else:
            raise ValueError("'dataset' variable must have value 'mnist' or 'cifar'")

    plot_general(acc_superposition, [], ['Superposition model', 'Baseline model'],
                 'Superposition vs. baseline model with ' + nn_cnn.upper() + ' model', 'Epoch', 'Accuracy (%)', [10], 0, 100)

    # plot_general([], acc_normal, ['Superposition model', 'Baseline model'],
    #                  'Superposition vs. baseline model with ' + nn_cnn.upper() + ' model', 'Epoch', 'Accuracy (%)', [10], 0, 100)

