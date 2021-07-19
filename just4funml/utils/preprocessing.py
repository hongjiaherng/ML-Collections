import numpy as np


def to_one_hot(y):
    
    if not isinstance(y, np.ndarray) or not y.ndim == 1:
        raise ValueError('Argument y must be a 1D numpy array')

    m = len(y)
    unique_classes = np.unique(y)
    Y_one_hot = np.zeros(shape=(m, unique_classes.size))

    for i in range(m):
        selected_col_index = np.argwhere(unique_classes == y[i])[0]
        Y_one_hot[i, selected_col_index] = 1.0

    return Y_one_hot