import numpy as np

def cost_function(type, Y, Y_hat):
    if type == "logistic":
        return logistic(Y, Y_hat)
    elif type == "multinomial_logistic":
        return multinomial_logistic(Y, Y_hat)
    elif type == "mse":
        return mean_squared_error(Y, Y_hat)
    else:
        raise Exception(f"{type} loss is not exist")

def loss_deriv(type, Y, Y_hat):
    if type == "logistic":
        return logistic_deriv(Y, Y_hat)
    elif type == "multinomial_logistic":
        return multinomial_logistic_deriv(Y, Y_hat)
    elif type == "mean_squared_error":
        return mean_squared_error_deriv(Y, Y_hat)
    else:
        raise Exception(f"{type} loss is not exist")

def logistic(Y, Y_hat): # Binary cross entropy loss / log loss
    m = Y_hat.shape[1]
    cost = -1 / m * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat), axis=1)
    
    return np.squeeze(cost)

def multinomial_logistic(Y, Y_hat): # Softmax loss / Categorical cross entropy
    cost = - np.sum(Y * np.log(Y_hat))
    
    return np.squeeze(cost)

def mean_squared_error(Y, Y_hat): # 2 * m is just for the sake of mathematical convenient when taking derivative
    m = Y_hat.shape
    cost = - 1 / (2*m) * np.sum(np.pow(Y_hat - Y, 2), axis=1)

    return np.squeeze(cost)

def logistic_deriv(Y, Y_hat):
    return (- Y / Y_hat) + ((1 - Y) / (1 - Y_hat))

def multinomial_logistic_deriv(Y, Y_hat):
    pass

def mean_squared_error_deriv(Y, Y_hat):
    pass