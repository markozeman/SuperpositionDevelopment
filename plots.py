import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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

