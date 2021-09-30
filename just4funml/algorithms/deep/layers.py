import numpy as np
from volume import Vol

class FlattenInput:
    def __init__(self, n_neurons):
        self.layer_type = "input"
        self.n_neurons = n_neurons
    
    def forward(self, vol_in):
        # print(self)
        self.vol_in = vol_in
        self.vol_out = vol_in
        
        return self.vol_out

    def backward(self):
        pass

class FullyConnected:
    def __init__(self, n_neurons):
        self.layer_type = "fc"
        self.n_neurons = n_neurons
        self.parameters = {}
    
    def forward(self, vol_in):
        # print(self)
        self.vol_in = vol_in
        # Compute Z = W . A_prev + b
        W = self.parameters["W"]
        b = self.parameters["b"]

        A_prev = vol_in.tensor
        Z = np.dot(A_prev, W) + b
        self.vol_out = Vol(Z)

        return self.vol_out
    
    def backward(self):
        # Compute dJdW, dJdb, dLdA_prev
        # Given dLdZ: self.vol_out.grads
        dLdZ = self.vol_out.grads # (m, n_this)
        dZdW = self.vol_in.tensor # (m, n_prev)
        
        m = dLdZ.shape[0]

        self.dJdW = 1 / m * np.dot(dZdW.T, dLdZ) # (n_prev, n_this) = (m, n_prev).T x (m, n_this)
        self.dJdb = 1 / m * np.sum(dLdZ, axis=0, keepdims=True) # (1, n_this)

        dLdA_prev = np.dot(dLdZ, self.parameters["W"].T) # (m, n_prev) = (m, n_this) x (n_prev, n_this).T
        self.vol_in.grads[:] = dLdA_prev

        # print(f"{self.layer_type} (out) \t-> {self.vol_out}")
        # print(f"{self.layer_type} (in) \t-> {self.vol_in}")
        



class ReLU:
    def __init__(self, n_neurons):
        self.layer_type = "relu"
        self.n_neurons = n_neurons

    def forward(self, vol_in):
        # print(self)
        self.vol_in = vol_in
        # Compute A = ReLU(vol_in)
        Z = vol_in.tensor
        A = np.maximum(0, Z)
        self.vol_out = Vol(A)

        return self.vol_out

    def backward(self):
        # Compute dLdZ
        # Given dLdA
        dLdA = self.vol_out.grads
        dAdZ = (self.vol_in.tensor > 0).astype("float") # same as self.vol_out.tensor 

        dLdZ = dLdA * dAdZ
        self.vol_in.grads[:] = dLdZ

        # print(f"{self.layer_type} (out) \t-> {self.vol_out}")
        # print(f"{self.layer_type} (in) \t-> {self.vol_in}")
        

class Sigmoid:
    def __init__(self, n_neurons):
        self.layer_type = "sigmoid"
        self.n_neurons = n_neurons

    def forward(self, vol_in):
        # print(self)
        self.vol_in = vol_in
        # Compute A = Sigmoid(vol_in)
        Z = vol_in.tensor
        A = 1 / (1 + np.exp(-Z))
        self.vol_out = Vol(A)

        return self.vol_out

    def backward(self):
        # Compute dLdZ, given dLdA
        dLdA = self.vol_out.grads
        dAdZ = self.vol_out.tensor * (1 - self.vol_out.tensor)
        dLdZ = dLdA * dAdZ
        self.vol_in.grads[:] = dLdZ

        # print(f"{self.layer_type} (out) \t-> {self.vol_out}")
        # print(f"{self.layer_type} (in) \t-> {self.vol_in}")
        

class Tanh:
    def __init__(self, n_neurons):
        self.layer_type = "tanh"
        self.n_neurons = n_neurons

    def forward(self, vol_in):
        # print(self)
        self.vol_in = vol_in
        # Compute A = tanh(vol_in)
        Z = vol_in.tensor
        A = (np.exp(2 * Z) - 1) / (np.exp(2 * Z) + 1)
        self.vol_out = Vol(A)

        return self.vol_out

    def backward(self):
        # Compute dLdZ, given dLdA
        dLdA = self.vol_out.grads
        dAdZ = 1 - np.power(self.vol_out.tensor, 2)
        dLdZ = dLdA * dAdZ
        self.vol_in.grads[:] = dLdZ

        # print(f"{self.layer_type} (out) \t-> {self.vol_out}")
        # print(f"{self.layer_type} (in) \t-> {self.vol_in}")


class Regression:
    def __init__(self, n_neurons):
        self.layer_type = "regression"
        self.n_neurons = n_neurons

    def forward(self, vol_in):
        # print(self)
        self.vol_in = vol_in
        self.vol_out = vol_in

        return self.vol_out

    def backward(self, Y):
        # y: (m, k)
        Y_hat = self.vol_in.tensor

        m = Y.shape[0]
        loss = 1 / (2 * m) * np.sum(np.power(Y_hat - Y, 2))

        dLdY_hat = Y_hat - Y
        self.vol_in.grads[:] = dLdY_hat
        
        # print(f"{self.layer_type} (out) \t-> {self.vol_out}")
        # print(f"{self.layer_type} (in) \t-> {self.vol_in}")
        
        return loss


class Logistic:
    def __init__(self):
        self.layer_type = "logistic"
        self.n_neurons = 1

    def forward(self, vol_in):
        # print(self)
        self.vol_in = vol_in
        self.vol_out = vol_in

        return self.vol_out

    def backward(self, Y):
        # Compute dLdA, given Y and Y_hat
        Y_hat = self.vol_in.tensor

        m = Y.shape[0]
        loss = -1 / m * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
        
        dLdY_hat = (-Y / Y_hat) + (1 - Y / 1 - Y_hat)
        self.vol_in.grads[:] = dLdY_hat

        # print(f"{self.layer_type} (out) \t-> {self.vol_out}")
        # print(f"{self.layer_type} (in) \t-> {self.vol_in}")

        return loss

class Softmax:
    def __init__(self, n_neurons):
        self.layer_type = "softmax"
        self.n_neurons = n_neurons
    
    def forward(self, vol_in):
        # print(self)
        self.vol_in = vol_in
        # Compute A = softmax(Z)
        Z = vol_in.tensor
        exp_z = np.exp(Z)
        A = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        self.vol_out = Vol(A)
        return self.vol_out

    def backward(self):
        pass
