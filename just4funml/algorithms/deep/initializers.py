import numpy as np

def weights_init(method, n_prev, n_this, seed):
    if method == "he":
        # Good for ReLU activation
        # W^[l] ~ N(0, 2/n^[l-1]) 
        std = np.sqrt(2 / n_prev)
    elif method == "xavier": 
        # Good for sigmoid, tanh activation
        # W^[l] ~ N(0, 1/n^[l-1])
        std = np.sqrt(1 / n_prev)
    elif method == "dunno_name":
        # Generally good
        # W^[l] ~ N(0, 2/(n^[l-1] + n^[l]))
        std = np.sqrt(2 / (n_prev + n_this))
    else:
        raise Exception(f"{method} initialization method doesn't exist")

    np.random.seed(seed)
    return np.random.randn(n_prev, n_this) * std

