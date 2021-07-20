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


def train_validation_test_split(X, y, validation_ratio=0.2, test_ratio=0.2, random_state=None):
    np.random.seed(random_state)
    
    total_size = len(X)

    test_size = int(total_size * test_ratio)
    validation_size = int(total_size * validation_ratio)
    train_size = total_size - test_size - validation_size

    rnd_indices = np.random.permutation(total_size)

    X_train = X[rnd_indices[:train_size]]
    y_train = y[rnd_indices[:train_size]]
    X_valid = X[rnd_indices[train_size:-test_size]]
    y_valid = y[rnd_indices[train_size:-test_size]]
    X_test = X[rnd_indices[-test_size:]]
    y_test = y[rnd_indices[-test_size:]]

    return X_train, y_train, X_valid, y_valid, X_test, y_test