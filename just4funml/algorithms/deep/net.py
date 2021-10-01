import numpy as np
import initializers
import layers
from volume import Vol

class Net:
    def __init__(self):
        self.params_init = False

    def bind_layers(self, layers_config):
        # Input checks
        assert len(layers_config) >= 2 # at least 2 layers
        assert layers_config[0]["layer_type"] == "input" # first layer must be input layer
        assert layers_config[-1]["layer_type"] == "softmax" or layers_config[-1]["layer_type"] == "regression" or layers_config[-1]["layer_type"] == "logistic" # last layer must be either regression or softmax

        # Parse layers_config to collect each independent layer info
        proper_layers_config = []

        for layer_info in layers_config:

            if layer_info["layer_type"] == "softmax" or layer_info["layer_type"] == "regression":
                proper_layers_config.append({"layer_type": "fc", "n_neurons": layer_info["n_neurons"]})
            elif layer_info["layer_type"] == "logistic":
                proper_layers_config.append({"layer_type": "fc", "n_neurons": 1})
                proper_layers_config.append({"layer_type": "sigmoid"})
                

            proper_layers_config.append(layer_info)   

            if layer_info["layer_type"] == "fc" and "activation" in layer_info:
                proper_layers_config.append({"layer_type": layer_info["activation"]})

        # Initialize the neural network's layers
        self.layers = []
        for i in range(len(proper_layers_config)):
            if proper_layers_config[i]["layer_type"] == "input": self.layers.append(layers.FlattenInput(proper_layers_config[i]["n_neurons"]))
            elif proper_layers_config[i]["layer_type"] == "fc": self.layers.append(layers.FullyConnected(proper_layers_config[i]["n_neurons"]))
            elif proper_layers_config[i]["layer_type"] == "relu": self.layers.append(layers.ReLU(proper_layers_config[i-1]["n_neurons"]))
            elif proper_layers_config[i]["layer_type"] == "sigmoid": self.layers.append(layers.Sigmoid(proper_layers_config[i-1]["n_neurons"]))
            elif proper_layers_config[i]["layer_type"] == "tanh": self.layers.append(layers.Tanh(proper_layers_config[i-1]["n_neurons"]))
            elif proper_layers_config[i]["layer_type"] == "softmax": self.layers.append(layers.Softmax(proper_layers_config[i]["n_neurons"]))
            elif proper_layers_config[i]["layer_type"] == "logistic": self.layers.append(layers.Logistic())
            elif proper_layers_config[i]["layer_type"] == "regression": self.layers.append(layers.Regression(proper_layers_config[i]["n_neurons"]))
            else: raise Exception(proper_layers_config[i]["layer_type"] + " layer doesn't exist")

    
    def init_params(self, method, seed=None):
        check_params = {}
        i = 1
        for l in range(len(self.layers)):
            if hasattr(self.layers[l], "parameters"):
                self.layers[l].parameters["W"] = initializers.weights_init(method=method, n_prev=self.layers[l-1].n_neurons, n_this=self.layers[l].n_neurons, seed=seed)
                self.layers[l].parameters["b"] = np.zeros((1, self.layers[l].n_neurons))
                check_params[f"W{i}"] = self.layers[l].parameters["W"]
                check_params[f"b{i}"] = self.layers[l].parameters["b"]
                i += 1
        self.params_init = True
        return check_params


    def forward_prop(self, X):
        assert self.params_init
        assert X.shape[1] == self.layers[0].n_neurons # dimension of input features X must match with n_neurons in first layer   

        X = Vol(X)
        act = self.layers[0].forward(X)

        for l in range(1, len(self.layers)):
            act = self.layers[l].forward(act)
        
        return act.tensor

    def backward_prop(self, Y):
        loss = self.layers[-1].backward(Y)
        for l in reversed(range(0, len(self.layers) - 1)):
            self.layers[l].backward()
        
        return loss
        