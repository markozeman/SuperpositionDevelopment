import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plot_accuracies_over_time(normal_accuracies, superposition_accuracies):
    """
    Plot accuracies of original test images over time with normal and superposition learning.

    :param normal_accuracies: list of accuracies with normal training
    :param superposition_accuracies: list of accuracies with superposition training
    :return: None
    """
    plt.plot(normal_accuracies)
    plt.plot(superposition_accuracies)
    plt.vlines(10, 0, 100, colors='red', linestyles='dotted')
    plt.legend(['Baseline model', 'Superposition model'])
    plt.title('Model accuracy with normal and superposition training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.show()


def plot_lr(learning_rates):
    """
    Plot changes of learning rate over time.

    :param learning_rates: list of learning rates
    :return: None
    """
    plt.plot(learning_rates)
    plt.title('Change of learning rate over time')
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.show()

