from collections import Counter
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
from dataset_preparation import get_dataset
from help_functions import permute_images
from plots import weights_heatmaps
from superposition import normal_training_mnist, nn, plot_confusion_matrix, get_context_matrices, \
    superposition_training_cifar, plot_general
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


def get_test_acc_for_individual_tasks(final_model, context_matrices, test_data, X_test):
    """
    Use superimposed 'final_model' to test the accuracy for each individual task when weights of the 'final_model'
    are unfolded according to 'context_matrices' and tested on appropriate 'test_data'.

    :param final_model: final model after superposition of all tasks
    :param context_matrices: multidimensional numpy array with random context (binary superposition)
    :param test_data: [(X_test_1, y_test_1), (X_test_2, y_test_2), ...]
    :param X_test: all test data
    :return: list of superposition accuracies for individual tasks, all tasks predictions,
    list of indices of incorrectly predicted samples for each task
    """
    all_task_predictions = []
    reverse_incorrect_indices = []
    reverse_accuracies = []
    num_of_tasks = len(test_data)
    for i in range(num_of_tasks - 1, -1, -1):   # go from the last task to the first one
        # test i-th task accuracy
        loss, accuracy = final_model.evaluate(test_data[i][0], test_data[i][1], verbose=2)
        reverse_accuracies.append(accuracy * 100)

        incorrect_indices = np.nonzero(final_model.predict_classes(test_data[i][0]).reshape((-1,)) != np.argmax(test_data[i][1], axis=1))[0]
        reverse_incorrect_indices.append(incorrect_indices)
        # print('incorrect:', len(incorrect_indices), incorrect_indices)

        # testing
        # task_predictions = final_model.predict(test_data[i][0])     # for different test data for every task (in case of 5 tasks of Permuting MNIST in function split_MNIST)
        task_predictions = final_model.predict(X_test)

        all_task_predictions.append(task_predictions)

        # weights_heatmaps([final_model.layers[2].get_weights()[0], final_model.layers[3].get_weights()[0]],
        #                  ['2-3 layer', '3-output layer'], i + 1)

        if i != 0:  # remain the weights for the first task
            # unfold weights one task back
            for layer_index, layer in enumerate(final_model.layers[1:]):  # first layer is Flatten so we skip it
                context_inverse_multiplied = np.linalg.inv(np.diag(context_matrices[i][layer_index]))    # matrix inverse is not necessary for binary context
                layer.set_weights([context_inverse_multiplied @ layer.get_weights()[0], layer.get_weights()[1]])

    return list(reversed(reverse_accuracies)), all_task_predictions, list(reversed(reverse_incorrect_indices))


def global_accuracy(task_accuracies, test_samples_per_task):
    """
    Calculate global accuracy of the model based on accuracies from single tasks accounting number of test samples per task.

    :param task_accuracies: list of accuracies for each task
    :param test_samples_per_task: list of test samples for each task
    :return: total/global accuracy in %
    """
    accurate_samples = sum([round((task_acc / 100) * task_samples) for task_acc, task_samples in zip(task_accuracies, test_samples_per_task)])
    return (accurate_samples / sum(test_samples_per_task)) * 100


def show_confusion_matrix(model, X_test, y_test):
    """
    Show confusion matrix based in the current model with test data.

    :param model: current model
    :param X_test: test input data
    :param y_test: test output labels
    :return: None
    """
    y_predicted = model.predict_classes(X_test)
    y_true = np.argmax(y_test, axis=1)      # reverse one-hot encoding
    conf_mat = confusion_matrix(y_true, y_predicted)
    plot_confusion_matrix(conf_mat)


def all_tasks_inference(all_tasks_predictions, y_test):
    """
    Compare inference of all_tasks_predictions with y_test.

    :param all_tasks_predictions: list of 2D arrays (length = num_of_tasks), each is consisted of len(y_test) x 10 (output),
                                  in reversed order (from last to the first task)
    :param y_test: test output labels (one-hot encoded)
    :return: None
    """
    i = 0
    incorrect = 0
    t5, t4, t3, t2, t1 = all_tasks_predictions
    for a, b, c, d, e in zip(t1, t2, t3, t4, t5):   # now right order of tasks
        max_index = np.argmax(np.concatenate((a, b, c, d, e)))
        max_index = max_index % 10

        # max_index = np.argmax(np.array(a) + np.array(b) + np.array(c) + np.array(d) + np.array(e))

        y = np.argmax(y_test[i])
        i += 1

        if max_index != y:
            incorrect += 1
            # print(max_index, y)
            # print(a)
            # print(b)
            # print(c)
            # print(d)
            # print(e)
            # print('\n')

    print('Incorrect examples:', incorrect)


def split_MNIST(model, X_train, y_train, X_test, y_test, input_size, num_of_units, num_of_tasks, num_of_epochs, batch_size):
    """
    Train split MNIST of 5 tasks ([0,1], ..., [8,9]) in superposition.

    :param model: Keras model instance
    :param X_train: train input data
    :param y_train: train output labels
    :param X_test: test input data
    :param y_test: test output labels
    :param input_size: image input size in pixels
    :param num_of_units: number of neurons in each hidden layer
    :param num_of_tasks: number of different tasks
    :param num_of_epochs: number of epochs for a task
    :param batch_size: batch size - number of samples per gradient update
    :return: None
    """
    '''
    # create 5 subtasks for MNIST ([0, 1], [2, 3], ..., [8, 9])
    d = []
    for i in range(0, 10, 2):
        X_train_extracted, y_train_extracted = extract_digits(X_train, y_train, [i, i + 1])  # extract only two digits
        X_test_extracted, y_test_extracted = extract_digits(X_test, y_test, [i, i + 1])  # extract only two digits
        d.append((X_train_extracted, y_train_extracted, X_test_extracted, y_test_extracted))
    '''


    # 5 tasks of Permuting MNIST with permuting X_test as well
    d = [(X_train, y_train, X_test, y_test)]
    for i in range(num_of_tasks - 1):
        d.append((permute_images(X_train, i), y_train, permute_images(X_test, i), y_test))


    context_matrices = get_context_matrices(input_size, num_of_units, num_of_tasks)
    acc_superposition = superposition_training_cifar(model, d, num_of_epochs, num_of_tasks, context_matrices, nn_cnn, batch_size)

    # plot_general(acc_superposition, [], ['Superposition model', 'Baseline model'],
    #              'Superposition vs. baseline model with ' + nn_cnn.upper() + ' model', 'Epoch', 'Accuracy (%)', [10], 0, 100)

    test_data = np.array(d, dtype=object)[:, 2:]

    task_identity_recognition(test_data, model, context_matrices)   # only for 5 tasks of Permuting MNIST

    sup_acc_per_task, all_tasks_predictions, _ = get_test_acc_for_individual_tasks(model, context_matrices, test_data, X_test)
    print('Accuracies for each task: ', sup_acc_per_task)
    # print('Global accuracy: ', global_accuracy(sup_acc_per_task, [2115, 2042, 1874, 1986, 1983]))  # for tasks in order [0,1] to [8,9]

    # show_confusion_matrix(model, X_test, y_test)

    all_tasks_inference(all_tasks_predictions, y_test)


def task_identity_recognition(test_data, final_model, context_matrices):
    """
    In 5 Permuting MNIST tasks check if original and differently permuted samples can infer the right task identity.

    :param test_data: [(X_test_1, y_test_1), (X_test_2, y_test_2), ...]
    :param final_model: final model after superposition of all tasks
    :param context_matrices: multidimensional numpy array with random context (binary superposition)
    :return: None
    """
    i = 4   # task ID
    for X_test, _ in [test_data[i]]:
        preds = [final_model.predict(X_test)]
        for index in range(4, 0, -1):
            for layer_index, layer in enumerate(final_model.layers[1:]):  # first layer is Flatten so we skip it
                context_inverse_multiplied = np.linalg.inv(np.diag(context_matrices[index][layer_index]))  # matrix inverse is not necessary for binary context
                layer.set_weights([context_inverse_multiplied @ layer.get_weights()[0], layer.get_weights()[1]])

            preds.append(final_model.predict(X_test))

        wrong = 0
        alll = 0
        for sample_number in range(10000):
            a = preds[4][sample_number]
            b = preds[3][sample_number]
            c = preds[2][sample_number]
            d = preds[1][sample_number]
            e = preds[0][sample_number]

            max_index = np.argmax(np.concatenate((a, b, c, d, e)))
            max_index = max_index // 10

            alll += 1

            if max_index != i:
                wrong += 1
                # print('i: ', i)
                # print('max index: ', max_index)

        print('\nwrong: ', wrong)
        print('all: ', alll)


def all_tasks_inference_one_vs_all(all_tasks_predictions, y_test):
    """
    Compare inference of all_tasks_predictions with y_test for one vs. all approach.

    :param all_tasks_predictions: list of 2D arrays (length = num_of_tasks), each is consisted of len(y_test) x 2 (output),
                                  in reversed order (from last to the first task)
    :param y_test: test output labels (one-hot encoded)
    :return: list of indices for incorrectly classified samples
    """
    index = 0
    incorrect = 0
    below_50 = 0
    below_50_correct = 0
    wrong_indices = []
    t10, t9, t8, t7, t6, t5, t4, t3, t2, t1 = all_tasks_predictions
    for a, b, c, d, e, f, g, h, i, j in zip(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10):   # now right order of tasks
        max_index = np.argmax(np.array([a[0], b[0], c[0], d[0], e[0], f[0], g[0], h[0], i[0], j[0]]))
        y = np.argmax(y_test[index])

        if max_index != y:  # incorrect prediction
            wrong_indices.append(index)
            # print('\nindex:', index)
            # print('real:', y, 'my pred.:', max_index)
            # print([a[0], b[0], c[0], d[0], e[0], f[0], g[0], h[0], i[0], j[0]])
            if [a[0], b[0], c[0], d[0], e[0], f[0], g[0], h[0], i[0], j[0]][max_index] < 0.5:
                below_50 += 1
            incorrect += 1
        else:   # correct prediction
            if [a[0], b[0], c[0], d[0], e[0], f[0], g[0], h[0], i[0], j[0]][max_index] < 0.5:
                below_50_correct += 1

        index += 1

    print('Incorrect examples:', incorrect)
    print('Incorrect examples < 50:', below_50)
    print('Correct examples < 50:', below_50_correct)
    return wrong_indices


def split_MNIST_one_vs_all(model, X_train, y_train, X_test, y_test, input_size, num_of_units, num_of_tasks, num_of_epochs, batch_size):
    """
    Train split MNIST of 10 tasks (each task is one vs. all) in superposition.

    :param model: Keras model instance
    :param X_train: train input data
    :param y_train: train output labels
    :param X_test: test input data
    :param y_test: test output labels
    :param input_size: image input size in pixels
    :param num_of_units: number of neurons in each hidden layer
    :param num_of_tasks: number of different tasks
    :param num_of_epochs: number of epochs for a task
    :param batch_size: batch size - number of samples per gradient update
    :return: None
    """
    # create 10 subtasks for MNIST
    d = []
    for i in range(10):
        y_train_extracted = [np.array([1, 0]) if np.argmax(label) == i else np.array([0, 1]) for label in y_train]
        y_test_extracted = [np.array([1, 0]) if np.argmax(label) == i else np.array([0, 1]) for label in y_test]
        d.append((X_train, np.array(y_train_extracted), X_test, np.array(y_test_extracted)))

    context_matrices = get_context_matrices(input_size, num_of_units, num_of_tasks)
    acc_superposition = superposition_training_cifar(model, d, num_of_epochs, num_of_tasks, context_matrices, nn_cnn, batch_size)

    plot_general(acc_superposition, [], ['Superposition model', 'Baseline model'],
                 'Superposition vs. baseline model with ' + nn_cnn.upper() + ' model', 'Epoch', 'Accuracy (%)', [10], 0, 100)

    test_data = np.array(d, dtype=object)[:, 2:]
    sup_acc_per_task, all_tasks_predictions, _ = get_test_acc_for_individual_tasks(model, context_matrices, test_data, X_test)
    print('Accuracies for each task: ', sup_acc_per_task)
    print('Global accuracy: ', global_accuracy(sup_acc_per_task,
          [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]))  # for tasks in numerical order

    # show_confusion_matrix(model, X_test, y_test)

    return all_tasks_inference_one_vs_all(all_tasks_predictions, y_test)


def all_tasks_inference_emsemble(all_tasks_predictions, y_test):
    """
    Compare inference of all_tasks_predictions with y_test for ensemble approach.

    :param all_tasks_predictions: list of 2D arrays (length = num_of_tasks), each is consisted of len(y_test) x 10 (output),
                                  in reversed order (from last to the first task)
    :param y_test: test output labels (one-hot encoded)
    :return: None
    """
    index = 0
    incorrect = 0
    t10, t9, t8, t7, t6, t5, t4, t3, t2, t1 = all_tasks_predictions
    # t5, t4, t3, t2, t1 = all_tasks_predictions
    for a, b, c, d, e, f, g, h, i, j in zip(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10):  # now right order of tasks
    # for a, b, c, d, e in zip(t1, t2, t3, t4, t5):  # now right order of tasks
        sum_up = a + b + c + d + e + f + g + h + i + j
        # sum_up = a + b + c + d + e

        a, b, c, d, e, f, g, h, i, j = np.argmax(a), np.argmax(b), np.argmax(c), np.argmax(d), np.argmax(e), \
                                       np.argmax(f), np.argmax(g), np.argmax(h), np.argmax(i), np.argmax(j)
        # a, b, c, d, e = np.argmax(a), np.argmax(b), np.argmax(c), np.argmax(d), np.argmax(e)

        counter = Counter([a, b, c, d, e, f, g, h, i, j])
        # counter = Counter([a, b, c, d, e])
        highest = max(counter, key=counter.get)

        y = np.argmax(y_test[index])

        if np.argmax(sum_up) != y:
            print('predictions', a, b, c, d, e, f, g, h, i, j)
            # print('predictions', a, b, c, d, e)
            print('pred sum up: ', np.argmax(sum_up))
            print('pred highest count: ', highest)
            print('y', y)
            print()
            incorrect += 1

        index += 1

    print('Incorrect examples:', incorrect)


def split_MNIST_ensemble(model, X_train, y_train, X_test, y_test, input_size, num_of_units, num_of_tasks, num_of_epochs, batch_size):
    """
    Train MNIST ensemble in superposition.

    :param model: Keras model instance
    :param X_train: train input data
    :param y_train: train output labels
    :param X_test: test input data
    :param y_test: test output labels
    :param input_size: image input size in pixels
    :param num_of_units: number of neurons in each hidden layer
    :param num_of_tasks: number of different tasks
    :param num_of_epochs: number of epochs for a task
    :param batch_size: batch size - number of samples per gradient update
    :return: None
    """
    # # to select only 66.32% random samples for each of the tasks
    # d = []
    # for _ in range(num_of_tasks):
    #     sampled_indices = np.array(list(map(round, np.random.uniform(-0.5, 9999.5, 6320))))
    #     d.append((X_train[sampled_indices], y_train[sampled_indices], X_test, y_test))

    # to select full train data for each task
    d = [(X_train, y_train, X_test, y_test) for _ in range(num_of_tasks)]

    context_matrices = get_context_matrices(input_size, num_of_units, num_of_tasks)
    acc_superposition = superposition_training_cifar(model, d, num_of_epochs, num_of_tasks, context_matrices, nn_cnn,
                                                     batch_size)

    plot_general(acc_superposition, [], ['Superposition model', 'Baseline model'],
                 'Superposition vs. baseline model with ' + nn_cnn.upper() + ' model', 'Epoch', 'Accuracy (%)', [10], 0, 100)

    test_data = np.array(d, dtype=object)[:, 2:]
    sup_acc_per_task, all_tasks_predictions, incorrect_indices = get_test_acc_for_individual_tasks(model, context_matrices, test_data, X_test)
    print('Accuracies for each task: ', sup_acc_per_task)
    print('Global accuracy: ', global_accuracy(sup_acc_per_task, [10000 for _ in range(num_of_tasks)]))

    intersection_indices = list(set.intersection(*map(set, incorrect_indices)))
    print(len(intersection_indices), intersection_indices)

    all_tasks_inference_emsemble(all_tasks_predictions, y_test)


if __name__ == '__main__':
    # to avoid cuDNN error (https://github.com/tensorflow/tensorflow/issues/24496)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    dataset = 'mnist'
    nn_cnn = 'nn'
    input_size = (28, 28)
    num_of_units = 1000
    num_of_classes = 10

    num_of_tasks = 5
    num_of_epochs = 10
    batch_size = 600

    X_train, y_train, X_test, y_test = get_dataset(dataset, nn_cnn, input_size, num_of_classes)
    model = nn(input_size, num_of_units, num_of_classes)

    split_MNIST(model, X_train, y_train, X_test, y_test, input_size, num_of_units, num_of_tasks, num_of_epochs, batch_size)

    # wrong_indices = split_MNIST_one_vs_all(model, X_train, y_train, X_test, y_test, input_size, num_of_units, num_of_tasks,
    #                        num_of_epochs, batch_size)
    # print('wrong_indices', len(wrong_indices), wrong_indices)

    # split_MNIST_ensemble(model, X_train, y_train, X_test, y_test, input_size, num_of_units, num_of_tasks, num_of_epochs, batch_size)

    '''
    X_train, y_train, X_test, y_test = get_dataset(dataset, nn_cnn, input_size, num_of_classes)
    model = nn(input_size, num_of_units, num_of_classes)
    num_of_tasks = 1
    num_of_epochs = 50
    acc_normal = normal_training_mnist(model, X_train, y_train, X_test, y_test, num_of_epochs, num_of_tasks, nn_cnn, batch_size)

    incorrect_indices = np.nonzero(model.predict_classes(X_test).reshape((-1,)) != np.argmax(y_test, axis=1))[0]
    print('incorrect_indices', len(incorrect_indices), incorrect_indices)

    _, accuracy = model.evaluate(X_test, y_test, verbose=2)
    print('accuracy:', accuracy * 100, (1 - (len(incorrect_indices) / 10000)) * 100)

    intersection = list(set(wrong_indices) & set(incorrect_indices))
    print('intersection', len(intersection), intersection)
    '''


