import copy
import pickle
import numpy as np
import tensorflow.keras.backend as K
from math import exp
# from keras.callbacks import Callback
from tensorflow.keras.callbacks import Callback
from scipy.stats import entropy


class TestPerformanceCallback(Callback):
    """
    Callback class for testing normal model performance at the beginning of every epoch.
    """
    def __init__(self, X_test, y_test, model):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.model = model  # this is only a reference, not a deep copy
        self.accuracies = []

    def on_epoch_begin(self, epoch, logs=None):
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        self.accuracies.append(accuracy * 100)


class TestSuperpositionPerformanceCallback(Callback):
    """
    Callback class for testing superposition NN model performance at the beginning of every epoch.
    """
    def __init__(self, X_test, y_test, context_matrices, model, task_index):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.context_matrices = context_matrices
        self.model = model  # this is only a reference, not a deep copy
        self.task_index = task_index
        self.accuracies = []
        self.LR = []    # list of changing learning rates through learning

    # def on_batch_begin(self, batch, logs=None):
    #     # Check the entropy of the batch and observe its trend.
    #     print('batch: ', batch)
    #     p = self.model.predict(self.X_test)
    #
    #     # sum entropy across all samples
    #     ent = entropy(p[3])
    #     print('ent:', ent)
    #     # print(p[:3])
    #     print()

    def lr_getter_Adam(self):
        """
        Get the value of the current learning rate with set Adam optimizer.

        Adopted from:
        https://stackoverflow.com/questions/48198031/how-to-add-variables-to-progress-bar-in-keras/48206009#48206009

        :return: current learning rate
        """
        # get values
        decay = self.model.optimizer.decay
        lr = self.model.optimizer.lr
        iters = self.model.optimizer.iterations     # only this variable should not be constant
        beta_1 = self.model.optimizer.beta_1
        beta_2 = self.model.optimizer.beta_2

        # calculate
        lr = lr * (1. / (1. + decay * K.cast(iters, K.dtype(decay))))
        t = K.cast(iters, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(beta_2, t)) / (1. - K.pow(beta_1, t)))
        return np.float32(K.eval(lr_t))

    def on_batch_begin(self, batch, logs=None):
        self.LR.append(self.lr_getter_Adam())
        # print('decay: ', self.model.optimizer.decay)
        # print('iters: ', self.model.optimizer.iterations)
        # print('beta_1: ', self.model.optimizer.beta_1)
        # print('beta_2: ', self.model.optimizer.beta_2)
        # print('\n')

    def on_epoch_begin(self, epoch, logs=None):
        if self.task_index == 0:    # first task (original MNIST images) - we did not use context yet
            loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=2)
            self.accuracies.append(accuracy * 100)
            return

        # save current model weights (without bias node)
        curr_w_matrices = []
        curr_bias_vectors = []
        for layer in self.model.layers[1:]:  # first layer is Flatten so we skip it
            curr_w_matrices.append(layer.get_weights()[0])
            curr_bias_vectors.append(layer.get_weights()[1])

        # temporarily change model weights to be suitable for first task (without bias node)
        for i, layer in enumerate(self.model.layers[1:]):  # first layer is Flatten so we skip it
            # not multiplying with inverse because inverse is the same in binary superposition with {-1, 1} on the diagonal
            # using only element-wise multiplication on diagonal vectors for speed-up
            context_inverse_multiplied = copy.deepcopy(self.context_matrices[self.task_index][i])
            for task_i in range(self.task_index - 1, 0, -1):
                context_inverse_multiplied = np.multiply(context_inverse_multiplied, self.context_matrices[task_i][i])

            '''
            # shuffle a part of context vector
            vector_size = len(context_inverse_multiplied)
            for iii in range(vector_size):
                percent_inverted = 50
                if iii % round(100 / percent_inverted) == 0:
                    context_inverse_multiplied[iii] = -1 * context_inverse_multiplied[iii]   # change bit
            '''

            context_inverse_multiplied = np.diag(context_inverse_multiplied)    # vector to diagonal matrix

            layer.set_weights([context_inverse_multiplied @ curr_w_matrices[i], curr_bias_vectors[i]])

        # evaluate accuracy
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        self.accuracies.append(accuracy * 100)

        # change model weights back (without bias node)
        for i, layer in enumerate(self.model.layers[1:]):  # first layer is Flatten so we skip it
            layer.set_weights([curr_w_matrices[i], curr_bias_vectors[i]])


class TestSuperpositionPerformanceCallback_CNN(Callback):
    """
    Callback class for testing superposition CNN model performance at the beginning of every epoch.
    """
    def __init__(self, X_test, y_test, context_matrices, model, task_index):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.context_matrices = context_matrices
        self.model = model  # this is only a reference, not a deep copy
        self.task_index = task_index
        self.accuracies = []

    def on_epoch_begin(self, epoch, logs=None):
        if self.task_index == 0:    # first task - we did not use context yet
            loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=2)
            self.accuracies.append(accuracy * 100)
            return

        # save current model weights (without bias node)
        curr_w_matrices = []
        curr_bias_vectors = []
        for i, layer in enumerate(self.model.layers):
            if i < 2 or i > 3:  # conv or dense layer
                curr_w_matrices.append(layer.get_weights()[0])
                curr_bias_vectors.append(layer.get_weights()[1])

        # temporarily change model weights to be suitable for first task (without bias node)
        for i, layer in enumerate(self.model.layers):
            if i < 2 or i > 3:  # conv or dense layer
                # not multiplying with inverse because inverse is the same in binary superposition with {-1, 1} on the diagonal
                # using only element-wise multiplication on diagonal vectors for speed-up

                if i < 2:  # conv layer
                    # flatten
                    context_vector = self.context_matrices[self.task_index][i]
                    for task_i in range(self.task_index - 1, 0, -1):
                        context_vector = np.multiply(context_vector, self.context_matrices[task_i][i])

                    new_w = np.reshape(np.multiply(curr_w_matrices[i].flatten(), context_vector), curr_w_matrices[i].shape)
                    layer.set_weights([new_w, curr_bias_vectors[i]])
                else:  # dense layer
                    context_inverse_multiplied = self.context_matrices[self.task_index][i - 2]
                    for task_i in range(self.task_index - 1, 0, -1):
                        context_inverse_multiplied = np.multiply(context_inverse_multiplied, self.context_matrices[task_i][i - 2])
                    context_inverse_multiplied = np.diag(context_inverse_multiplied)    # vector to diagonal matrix

                    layer.set_weights([context_inverse_multiplied @ curr_w_matrices[i - 2], curr_bias_vectors[i - 2]])

        # evaluate accuracy
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        self.accuracies.append(accuracy * 100)

        # change model weights back (without bias node)
        for i, layer in enumerate(self.model.layers):
            if i < 2 or i > 3:  # conv or dense layer
                if i < 2:  # conv layer
                    layer.set_weights([curr_w_matrices[i], curr_bias_vectors[i]])
                else:  # dense layer
                    layer.set_weights([curr_w_matrices[i - 2], curr_bias_vectors[i - 2]])


lr_over_time = []   # global variable to store changing learning rates


def lr_scheduler(epoch, lr):
    """
    Learning rate scheduler function to set how learning rate changes each epoch.

    :param epoch: current epoch number
    :param lr: current learning rate
    :return: new learning rate
    """
    num_of_epochs = 10  # todo

    global lr_over_time
    lr_over_time.append(lr)

    decay_type = 'exponential'
    if decay_type == 'linear':
        lr -= 10 ** -5
    elif decay_type == 'exponential':
        initial_lr = 0.0001
        k = 0.07
        t = len(lr_over_time)
        lr = initial_lr * exp(-k * t)

    if len(lr_over_time) % num_of_epochs == 0:    # to start each new task with the same learning rate as the first one
        lr_over_time = []   # re-initiate learning rate

    return max(lr, 0.000001)    # don't let learning rate go to 0


class PrintDiscreteAccuracy(Callback):
    """
    Callback class for printing discrete context accuracy while training (after every epoch).
    Model has additional layers for factor -1 or 1 transformation.
    """
    def __init__(self, X_test, y_test, model, context_matrices):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.model = model  # this is only a reference, not a deep copy
        self.context_matrices = context_matrices
        self.starting_context_values = np.array([[1 if x > 0 else -1 for x in self.model.layers[2].get_weights()[0]],
                                                 [1 if x > 0 else -1 for x in self.model.layers[4].get_weights()[0]],
                                                 [1 if x > 0 else -1 for x in self.model.layers[6].get_weights()[0]]])
        self.last_context_values = self.context_matrices[0]     # start with random context values

    def on_epoch_end(self, epoch, logs=None):
        # save current custom layers' weights
        l_2_old = self.model.layers[2].get_weights()
        l_4_old = self.model.layers[4].get_weights()
        l_6_old = self.model.layers[6].get_weights()

        # discretize weights in custom layers
        l_2 = [np.array([1 if x > 0 else -1 for x in self.model.layers[2].get_weights()[0]])]
        l_4 = [np.array([1 if x > 0 else -1 for x in self.model.layers[4].get_weights()[0]])]
        l_6 = [np.array([1 if x > 0 else -1 for x in self.model.layers[6].get_weights()[0]])]
        self.model.layers[2].set_weights(l_2)
        self.model.layers[4].set_weights(l_4)
        self.model.layers[6].set_weights(l_6)

        res = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print("Discrete test accuracy (%):", round(res[1] * 100, 2))

        # compare context between context_matrices and the learned contexts over all layers
        learned_contexts = [l_2, l_4, l_6]
        count = {'0': 0, '1': 0, '2': 0}
        count_context_epoch_change = {'0': 0, '1': 0, '2': 0}
        for ind in range(3):
            # context_matrices[0], because we multiply with row 0 of context matrices in the method
            # context_matrices[1], because we use the second ([1]) index to go from the first to the second task
            # for a, b in zip(self.context_matrices[1][ind], learned_contexts[ind][0]):
            #     if a != b:
            #         count[str(ind)] += 1

            # count how many bits of contexts changed in each layer from the last epoch
            for a, b in zip(self.last_context_values[ind], learned_contexts[ind][0]):
                if a != b:
                    count_context_epoch_change[str(ind)] += 1

        if epoch != 0:
            print('Context bit changes in each layer from the last epoch: ', count_context_epoch_change)
        # print('Different context values count: ', count, '\n')

        # update the context to the current epoch
        self.last_context_values = np.array([l_2[0], l_4[0], l_6[0]])

        # save last contexts to use it later
        # pickle.dump(self.last_context_values, open('temp_learned_contexts_30_newnew.pkl', 'wb'))

        # set weights back to the pre-evaluation state
        self.model.layers[2].set_weights(l_2_old)
        self.model.layers[4].set_weights(l_4_old)
        self.model.layers[6].set_weights(l_6_old)

