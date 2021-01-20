import math
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def show_grayscale_image(img):
    """
    Show grayscale image (1 channel).

    :param img: image to plot
    :return: None
    """
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


def show_image(img):
    """
    Show coloured image (3 channels).

    :param img: image to plot
    :return: None
    """
    plt.imshow(img)
    plt.show()


def plot_general(line_1, line_2, legend_lst, title, x_label, y_label, vertical_lines_x, vl_min, vl_max, text_strings=None):
    """
    Plot two lines on the same plot with additional general information.

    :param line_1: y values of the first line
    :param line_2: y values of the second line
    :param legend_lst: list of two values -> [first line label, second line label]
    :param title: plot title (string)
    :param x_label: label of axis x (string)
    :param y_label: label of axis y (string)
    :param vertical_lines_x: x values of where to draw vertical lines
    :param vl_min: vertical lines minimum y value
    :param vl_max: vertical lines maximum y value
    :param text_strings: optional list of text strings to add to the bottom of vertical lines
    :return: None
    """
    font = {'size': 18}
    plt.rc('font', **font)

    plt.plot(line_1, linewidth=3)
    plt.plot(line_2, linewidth=3)
    plt.vlines(vertical_lines_x, vl_min, vl_max, colors='k', alpha=0.5, linestyles='dotted', linewidth=3)
    plt.legend(legend_lst)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if text_strings is not None:
        for i in range(len(text_strings)):
            plt.text(vertical_lines_x[i] + 0.25, vl_min, text_strings[i], colors='k', alpha=0.5)
    plt.show()


def plot_many_lines(lines, legend, title, x_label, y_label):
    """
    Plot many lines (of the same length) on the x-axis.

    :param lines: list of lists of values
    :param legend: label for each line (len(lines) = len(legend))
    param title: plot title (string)
    :param x_label: label of axis x (string)
    :param y_label: label of axis y (string)
    :return: None
    """
    for l in lines:
        plt.plot(l)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(legend)
    plt.show()


def plot_weights_histogram(x, bins):
    """
    Plot weights values on histogram.

    :param x: data/values to plot
    :param bins: number of bins on histogram
    :return: None
    """
    plt.hist(x, bins=bins)
    plt.title('Values of trained weights in the network')
    plt.xlabel('Weight value')
    plt.ylabel('Occurrences')
    plt.show()


def weights_heatmaps(W_matrices, labels, task_index):
    """
    Plot heat maps of weights from layers in the network.

    :param W_matrices: list of 2D numpy arrays which represent weights between layers
    :param labels: list of strings to put in the plot title
    :param task_index: integer index of the current task
    :return: None
    """
    # norm_matrix = (W_matrix - np.min(W_matrix)) / np.ptp(W_matrix)   # normalise matrix between [0, 1]
    plt.figure()
    if len(W_matrices) <= 3:
        plot_layout = (1, len(W_matrices))
    else:
        plot_layout = (2, math.ceil(len(W_matrices) / 2))

    for layer_index, weights_matrix in enumerate(W_matrices):
        plt.subplot(*plot_layout, layer_index + 1)
        sns.heatmap(weights_matrix, cmap='coolwarm', linewidth=0) if layer_index < 2 or layer_index > 3 \
                    else sns.heatmap(weights_matrix, cmap='Blues', linewidth=0)
        plt.title("Task %d || %s" % (task_index, labels[layer_index]))
    plt.tight_layout()
    plt.show()
    # plt.savefig('../../../Plots/Reproducible results/splitMNIST/20units_50epochs/W heatmaps/plot_%s.png' % str(task_index), bbox_inches='tight', dpi=300)


def plot_confusion_matrix(conf_mat):
    """
    Plot already calculated confusion matrix.

    :param conf_mat: 2D confusion matrix
    :return: None
    """
    # compute accuracy
    all_cases = conf_mat.sum()
    correct_cases = conf_mat.diagonal().sum()
    acc = (correct_cases / all_cases) * 100

    size = conf_mat.shape[0]
    df_cm = pd.DataFrame(conf_mat, range(size), range(size))
    sns.heatmap(df_cm, annot=True, cmap='Blues', linewidth=0, fmt='d')
    plt.title('Test accuracy: %d / %d = %.2f %%' % (correct_cases, all_cases, acc))
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.show()


def plot_multiple_results(data, legend_lst, colors, x_label, y_label, vertical_lines_x, vl_min, vl_max, show_CI=True, text_strings=None):
    """
    Plot more lines from the saved results on the same plot with additional information.

    :param data: list of 2D matrices, each matrix has more samples of the same experiment (number of experiments x length of experiment)
    :param legend_lst: list of label values (length of data)
    :param colors: list of colors used for lines (length of data)
    :param x_label: label of axis x (string)
    :param y_label: label of axis y (string)
    :param vertical_lines_x: x values of where to draw vertical lines
    :param vl_min: vertical lines minimum y value
    :param vl_max: vertical lines maximum y value
    :param show_CI: show confidence interval range (boolean)
    :param text_strings: optional list of text strings to add to the bottom of vertical lines
    :return: None
    """
    # font = {'size': 20}
    # plt.rc('font', **font)

    # plot lines with confidence intervals
    for i, data in enumerate(data):
        matrix = np.array(data)
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)

        # take only every n-th element of the array
        n = 1
        mean = mean[0::n]
        std = std[0::n]

        # plot the shaded range of the confidence intervals (mean +/- 2*std)
        if show_CI:
            up_limit = mean + (2 * std)
            up_limit[up_limit > 100] = 100  # cut accuracies above 100
            down_limit = mean - (2 * std)
            plt.fill_between(range(0, mean.shape[0] * n, n), up_limit, down_limit, color=colors[i], alpha=0.25)

        # plot the mean on top (every other line is dashed)
        if i % 2 == 0:
            plt.plot(range(0, mean.shape[0] * n, n), mean, colors[i], linewidth=3)
        else:
            plt.plot(range(0, mean.shape[0] * n, n), mean, colors[i], linewidth=3, linestyle='--')

    if legend_lst:
        plt.legend(legend_lst)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.vlines(vertical_lines_x, vl_min, vl_max, colors='k', linestyles='dashed', linewidth=2, alpha=0.5)
    if text_strings is not None:
        for i in range(len(text_strings)):
            plt.text(vertical_lines_x[i] + 0.5, vl_min, text_strings[i], color='k', alpha=0.5)
    plt.show()


