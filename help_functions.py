import copy

import numpy as np

from plots import plot_many_lines

seeds = [(i * 7) + 1 for i in range(3000)]    # random seeds for permutations, but remain the same each run (range is the maximum number of tasks)


def permute_pixels(im, seed):
    """
    Randomly permute pixels of image 'im'.

    :param im: image to be permuted (2D numpy array)
    :param seed: number that serves to have the same permutation for all images in the array
    :return: permuted image (2D numpy array)
    """
    im_1d = im.flatten()
    im_1d_permuted = np.random.RandomState(seed=seed).permutation(im_1d)
    return np.reshape(im_1d_permuted, im.shape)


def permute_images(images, permutation_index):
    """
    Permute pixels in all images.

    :param images: numpy array of images
    :param permutation_index: index of the permutation (#permutations = #tasks - 1)
    :return: numpy array of permuted images (of the same size)
    """
    # seed = np.random.randint(low=4294967295, dtype=np.uint32)    # make a random seed for all images in an array

    # baseline and superposition have the same permutation of images for the corresponding task
    global seeds
    seed = seeds[permutation_index]     # the same permutation each run for the first, second, ... task
    return np.array([permute_pixels(im, seed) for im in images])


def random_binary_array(size, task_index, layer_index):
    """
    Create an array of 'size' length consisting only of numbers -1 and 1 (approximately 50% each).

    :param size: shape of the created array
    :param task_index: index of a task (in reality task_index=0 means the second task since the first does not have context)
    :param layer_index: index of the layer (1 for the input layer etc.)
    :return: binary numpy array with values -1 or 1
    """
    # to make sure that each task in each layer has a different seed (but seeds are the same for different runs)
    global seeds
    seed = seeds[task_index] + layer_index
    np.random.seed(seed)

    # np.random.seed(1)   # set fixed seed to have always the same random vectors
    vec = np.random.uniform(-1, 1, size)
    vec[vec < 0] = -1
    vec[vec >= 0] = 1
    return vec


def get_context_matrices(input_size, num_of_units, num_of_tasks):
    """
    Get random context matrices for simple neural network that uses binary superposition as a context.

    :param input_size: image input size in pixels
    :param num_of_units: number of neurons in each hidden layer
    :param num_of_tasks: number of different tasks (permutations of original images)
    :return: multidimensional numpy array with random context (binary superposition)
    """
    context_matrices = []
    for i in range(num_of_tasks):
        C1 = random_binary_array(input_size[0] * input_size[1], i, 1)
        C2 = random_binary_array(num_of_units, i, 2)
        C3 = random_binary_array(num_of_units, i, 3)
        context_matrices.append([C1, C2, C3])

    context_stats(context_matrices)

    return context_matrices


def get_context_matrices_CNN(model, num_of_tasks):
    """
    Get random context matrices for simple convolutional neural network that uses binary superposition as a context.

    :param model: Keras model instance
    :param num_of_tasks: number of different tasks
    :return: multidimensional numpy array with random context (binary superposition)
    """
    context_shapes = []
    for i, layer in enumerate(model.layers):
        if i < 2 or i > 3:   # conv layer or dense layer
            context_shapes.append(layer.get_weights()[0].shape)

    context_matrices = []
    for i in range(num_of_tasks):
        _, kernel_size, tensor_width, num_of_conv_layers = context_shapes[0]
        C1 = random_binary_array(kernel_size * kernel_size * tensor_width * num_of_conv_layers, i, 1)   # conv layer
        _, kernel_size, tensor_width, num_of_conv_layers = context_shapes[1]
        C2 = random_binary_array(kernel_size * kernel_size * tensor_width * num_of_conv_layers, i, 2)   # conv layer
        C3 = random_binary_array(context_shapes[2][0], i, 3)  # dense layer
        C4 = random_binary_array(context_shapes[3][0], i, 4)  # dense layer

        '''
        # fixed context initialization (10 tasks, each has 10% of layer vectors -1 at different positions)
        l_1 = 86
        l_2 = 1843
        l_3 = 1254
        l_4 = 100

        _, kernel_size, tensor_width, num_of_conv_layers = context_shapes[0]
        C1 = np.full(shape=kernel_size * kernel_size * tensor_width * num_of_conv_layers, fill_value=1)
        C1[i * l_1: (i + 1) * l_1] = np.full(shape=l_1, fill_value=-1)

        _, kernel_size, tensor_width, num_of_conv_layers = context_shapes[1]
        C2 = np.full(shape=kernel_size * kernel_size * tensor_width * num_of_conv_layers, fill_value=1)
        C2[i * l_2: (i + 1) * l_2] = np.full(shape=l_2, fill_value=-1)

        C3 = np.full(shape=context_shapes[2][0], fill_value=1)
        C3[i * l_3: (i + 1) * l_3] = np.full(shape=l_3, fill_value=-1)

        C4 = np.full(shape=context_shapes[3][0], fill_value=1)
        C4[i * l_4: (i + 1) * l_4] = np.full(shape=l_4, fill_value=-1)
        '''

        context_matrices.append([C1, C2, C3, C4])
    return context_matrices


def context_stats(context_matrices):
    """
    Display statistics of context matrices in terms of dot product (for MNIST or 3 context matrices).

    :param context_matrices: multidimensional numpy array with random context (binary superposition)
    :return: None
    """
    # leave first row out since this context is not used
    stats_l1 = layer_ortho_stats(np.array(context_matrices)[1:, 0])
    stats_l2 = layer_ortho_stats(np.array(context_matrices)[1:, 1])
    stats_l3 = layer_ortho_stats(np.array(context_matrices)[1:, 2])

    plot_many_lines([stats_l1[:, 0], stats_l2[:, 0], stats_l3[:, 0], stats_l1[:, 1], stats_l2[:, 1], stats_l3[:, 1]],
                    ['L1 sum', 'L2 sum', 'L3 sum', 'L1 mean', 'L2 mean', 'L3 mean'],
                    'Size L1 = %d,   Size L2 = %d,   Size L3 = %d' % (len(context_matrices[0][0]), len(context_matrices[0][1]), len(context_matrices[0][2])),
                    'task number - 2', 'orthogonality')


def layer_ortho_stats(layer_cv):
    """
    Compute dot product stats for a specific layer.

    :param layer_cv: vector of context vectors, for a specific layer all task contexts from  the first one on
    :return: 2D list, first dimension are tasks from the second on (where comparison starts to be possible),
    the second dimension is a tuple of (normalized sum of dot products to the first vector, normalized average sum of dot products to the first vector,
    list of dot products between current context vector and all back to the first one)
    """
    all_orthos = []
    for index in range(len(layer_cv)):
        if index > 0:
            curr_vec = layer_cv[index]
            vec_orthos = []
            for i in range(index-1, -1, -1):
                vec_orthos.append(np.dot(curr_vec, layer_cv[i]))
            sum_orthos_normalized = sum([abs(v_o) for v_o in vec_orthos]) / len(curr_vec) * 100  # * 100 to increase small values (away from 0)
            all_orthos.append((sum_orthos_normalized, round(sum_orthos_normalized / len(vec_orthos), 1), vec_orthos))
    return np.array(all_orthos)


def context_multiplication(model, context_matrices, task_index):
    """
    Multiply current model weights with context matrices in each layer (without changing weights from bias node).

    :param model: Keras model instance
    :param context_matrices: multidimensional numpy array with random context (binary superposition)
    :param task_index: index of a task to know which context_matrices row to use
    :return: None (but model weights are changed)
    """
    for i, layer in enumerate(model.layers[1:]):  # first layer is Flatten so we skip it
        curr_w = layer.get_weights()[0]
        curr_w_bias = layer.get_weights()[1]

        new_w = np.diag(context_matrices[task_index][i]) @ curr_w
        layer.set_weights([new_w, curr_w_bias])


def context_multiplication_CNN(model, context_matrices, task_index):
    """
    Multiply current model weights in CNN with context matrices in each layer (without changing weights from bias node).

    :param model: Keras model instance
    :param context_matrices: multidimensional numpy array with random context (binary superposition)
    :param task_index: index of a task to know which context_matrices row to use
    :return: None (but model weights are changed)
    """
    for i, layer in enumerate(model.layers):
        if i < 2 or i > 3:  # conv or dense layer
            curr_w = layer.get_weights()[0]
            curr_w_bias = layer.get_weights()[1]

            if i < 2:   # conv layer
                new_w = np.reshape(np.multiply(curr_w.flatten(), context_matrices[task_index][i]), curr_w.shape)
            else:    # dense layer
                new_w = np.diag(context_matrices[task_index][i - 2]) @ curr_w  # -2 because of Flatten and MaxPooling layers

            layer.set_weights([new_w, curr_w_bias])


def print_validation_acc(history, task_index):
    """
    Print validation accuracy over epochs.

    :param history: Keras History object
    :param task_index: index of a task to know which context_matrices row to use
    :return: None
    """
    val_acc = np.array(history.history['val_accuracy']) * 100
    print('\nValidation accuracies: i =', task_index, val_acc)


def zero_out_vector(vec, proportion_0):
    """
    Zero out 'proportion_0' values in vector 'vec' with the lowest absolute magnitude.

    :param vec: vector of numeric values (numpy array)
    :param proportion_0: share of zeros we want in vector 'vec' (value between 0 and 1)
    :return: new vector with specified proportion of 0
    """
    vec_sorted = sorted(np.absolute(vec))
    abs_threshold = vec_sorted[round(len(vec) * proportion_0)]
    mask = (np.absolute(vec) > abs_threshold).astype(float)
    return mask * vec

