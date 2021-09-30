import numpy as np

def non_linearity(Z, type):
    if type == "sigmoid":
        return sigmoid(Z)
    elif type == "relu":
        return relu(Z)
    elif type == "tanh":
        return tanh(Z)
    elif type == "softmax":
        pass
    else:
        raise Exception(f"{type} activation is not exist")
    
def non_linearity_deriv(Z, type):
    if type == "sigmoid":
        return sigmoid_deriv(Z)
    elif type == "relu":
        return relu_deriv(Z)
    elif type == "tanh":
        return tanh_deriv(Z)
    elif type == "softmax":
        pass
    else:
        raise Exception(f"{type} activation is not exist")

def sigmoid(Z):
    cache = (Z, "sigmoid")
    A = 1 / (1 + np.exp(-Z))
    return A, cache

def relu(Z):
    cache = (Z, "relu")
    A = np.maximum(0, Z)
    return A, cache

def tanh(Z):
    cache = (Z, "tanh")
    A = (np.exp(2 * Z) - 1) / (np.exp(2 * Z) + 1)
    return A, cache

def softmax(Z):
    cache = (Z, "softmax")
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A, cache

def sigmoid_deriv(Z):
    return (1 / (1 + np.exp(-Z))) * (1 - (1 / (1 + np.exp(-Z))))

def relu_deriv(Z):
    return (Z > 0).astype('float') # Assume undefined point at Z = 0, its derivative is 0

def tanh_deriv(Z):
    return 1 - np.pow((np.exp(2 * Z) - 1) / (np.exp(2 * Z) + 1), 2)

def softmax_deriv(Z): # shape of Z (K, m)
    pass
