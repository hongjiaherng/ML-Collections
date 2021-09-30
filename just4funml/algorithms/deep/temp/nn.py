import numpy as np
from sklearn import datasets
import activation_helper
import cost_helper

class FCNeuralNetwork:
    def __init__(self, activations, layer_dims):
        self.activations = activations
        self.n_layers = len(activations)
        self.layer_dims = layer_dims

    def init_parameters(self, method):
        parameters = {}

        for l in range(1, len(self.layer_dims)):
            if method == "xavier": # Good for sigmoid, tanh activation
                # W^[l] ~ N(0, 1/n^[l-1])
                std = np.sqrt(1 / self.layer_dims[l-1])
            elif method == "he": # Good for ReLU activation
                # W^[l] ~ N(0, 2/n^[l-1]) 
                std = np.sqrt(2 / self.layer_dims[l-1])
            else: 
                # W^[l] ~ N(0, 2/(n^[l-1] + n^[l]))
                std = np.sqrt(2 / (self.layer_dims[l-1] + self.layer_dims[l]))

            parameters[f"W{l}"] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * std
            parameters[f"b{l}"] = np.zeros((self.layer_dims[l], 1))
        
        return parameters

    def compute_cost(self, loss, Y_hat, Y):
        cost = cost_helper.cost_function(type=loss, Y=Y, Y_hat=Y_hat)
        return cost

    def forward_prop(self, X, parameters):
        caches = []
        A = X

        for l in range(self.n_layers):
            A_prev = A
            Z, linear_cache = self.linear_forward(A_prev, parameters[f"W{l+1}"], parameters[f"b{l+1}"])
            A, activation_cache = self.activation_forward(Z, activation=self.activations[l])

            combined_cache = (linear_cache, activation_cache)
            caches.append(combined_cache)

        return A, caches
    
    def backward_prop(self, loss, Y_hat, Y, caches):
        grads = {}

        dLdA_prev = cost_helper.loss_deriv(type=loss, Y=Y, Y_hat=Y_hat)

        for l in reversed(range(1, self.n_layers + 1)):
            current_layer_cache = caches[l - 1]
            linear_cache = current_layer_cache[0]
            activation_cache = current_layer_cache[1]

            dLdZ = self.activation_backward(dLdA=dLdA_prev, cache=activation_cache)
            dLdA_prev, dJdW, dJdb = self.linear_backward(dLdZ=dLdZ, cache=linear_cache)
            
            grads[f"dJdW{l}"] = dJdW
            grads[f"dJdb{l}"] = dJdb
            grads[f"dLdA{l-1}"] = dLdA_prev
        
        return grads
        
    def update_parameters_with_gd(self, parameters, grads, learning_rate):
        params = parameters.copy() # shallow copy

        for l in range(1, self.n_layers + 1):
            params[f"W{l}"] = params[f"W{l}"] - learning_rate * grads[f"dJdW{l}"]
            params[f"b{l}"] = params[f"b{l}"] - learning_rate * grads[f"dJdb{l}"]
        return params

    def train(self, X, Y, learning_rate, epochs, param_init, loss, verbose):
        
        parameters = self.init_parameters(method=param_init)
        parameters, costs = self.gradient_descent(X, Y, parameters, learning_rate, epochs, loss, verbose)
        return parameters, costs
    
    def gradient_descent(self, X, Y, parameters, learning_rate, epochs, loss, verbose):
        costs = []
        for i in range(epochs):
            Y_hat, caches = self.forward_prop(X, parameters)
            cost = self.compute_cost(loss, Y_hat, Y)
            grads = self.backward_prop(loss, Y_hat, Y, caches)
            parameters = self.update_parameters_with_gd(parameters, grads, learning_rate)

            if verbose and i % 100 == 0 or i == epochs - 1:
                print(f"Cost after epoch {i}: {cost}")
            if i % 100 == 0 or i == epochs - 1:
                costs.append(cost)

        return parameters, costs

    def predict(self, X, parameters):
        Y_pred = np.zeros((1, X.shape[1]))
        Y_hat, _ = self.forward_prop(X, parameters)

        for i in range(0, Y_hat.shape[1]):
            if Y_hat[0, i] > 0.5:
                Y_pred[0, i] = 1
            else:
                Y_pred[0, i] = 0

        return Y_pred

    @staticmethod
    def linear_forward(A_prev, W, b):
        Z = np.dot(W, A_prev) + b
        cache = (A_prev, W, b)
        return Z, cache

    @staticmethod
    def activation_forward(Z, activation):
        A, cache = activation_helper.non_linearity(Z, type=activation.lower())
        
        return A, cache

    @staticmethod
    def linear_backward(dLdZ, cache):
        A_prev, W, _ = cache
        m = A_prev.shape[1]

        dJdW = 1 / m * np.dot(dLdZ, A_prev.T) # shape (n^[l], n^[l-1])
        dJdb = 1 / m * np.sum(dLdZ, axis=1, keepdims=True) # shape (n^[l], 1)
        dLdA_prev = np.dot(W.T, dLdZ) # shape (n^[l-1], m)

        return dLdA_prev, dJdW, dJdb

    @staticmethod
    def activation_backward(dLdA, cache):
        Z, activation = cache
        
        dAdZ = activation_helper.non_linearity_deriv(Z, type=activation.lower())
        dLdZ = dLdA * dAdZ

        return dLdZ

from sklearn.datasets import load_iris

iris_dataset = load_iris()
X = iris_dataset["data"]
y = iris_dataset["target"]
X_processed = X[:, 2:].T
Y_processed = (y == 0).astype('int') 
Y_processed = Y_processed[np.newaxis, :]

nn = FCNeuralNetwork(activations=["relu", "sigmoid"], layer_dims=[2, 10, 1])
parameters, costs = nn.train(X_processed, Y_processed, learning_rate=0.01, epochs=1000, param_init="he", loss="logistic", verbose=True)
Y_predict = nn.predict(X_processed, parameters)
print(Y_processed)
print(Y_predict)
print(np.sum((Y_predict == Y_processed)) / X_processed.shape[1])