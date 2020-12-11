import numpy as np


def preprocess_SEA(path):
    """
    Preprocess saved SEA concept data from the given path to be ready for NN training.

    :param path: file path to the data
    :return: (X, y) as numpy arrays
    """
    with open(path) as f:
        lines = f.readlines()

    X = []
    y = []
    for line in lines:
        first, second, third, label = line.strip().split(',')
        X.append([[float(first), float(second), float(third)]])

        label = [1, 0] if label == '0' else [0, 1]
        y.append(label)

    return np.array(X), np.array(y)


def preprocess_Stagger(path):
    """
    Preprocess saved Stagger data from the given path to be ready for NN training.

    :param path: file path to the data
    :return: (X, y) as numpy arrays
    """
    d = {
        'size': {
            'small': [1, 0, 0],
            'medium': [0, 1, 0],
            'large': [0, 0, 1]
        },
        'color': {
            'red': [1, 0],
            'green': [0, 1]
        },
        'shape': {
            'circular': [1, 0],
            'non-circular': [0, 1]
        }
    }

    with open(path) as f:
        lines = f.readlines()

    X = []
    y = []
    for i, line in enumerate(lines):
        if i >= 7:
            size, color, shape, label = line.strip().split(',')
            X.append([d['size'][size] + d['color'][color] + d['shape'][shape]])

            label = [1, 0] if label == 'n' else [0, 1]
            y.append(label)

    return np.array(X), np.array(y)


if __name__ == '__main__':
    sea_path = 'data/SEA concept/SEA_60k.data'
    X_sea, y_sea = preprocess_SEA(sea_path)

    stagger_path = 'data/Stagger/stagger_w_50_n_0.1_101.arff'
    X_stagger, y_stagger = preprocess_Stagger(stagger_path)


