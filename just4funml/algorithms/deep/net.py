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

    
    def init_params(self, method):
        for l in range(len(self.layers)):
            if hasattr(self.layers[l], "parameters"):
                self.layers[l].parameters["W"] = initializers.weights_init(method=method, n_prev=self.layers[l-1].n_neurons, n_this=self.layers[l].n_neurons)
                self.layers[l].parameters["b"] = np.zeros((1, self.layers[l].n_neurons))
        self.params_init = True
        

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
        

layers_config = []
# layers_config.append({"layer_type": "input", "n_neurons": 2})
# layers_config.append({"layer_type": "fc", "n_neurons": 100, "activation": "relu"})
# layers_config.append({"layer_type": "fc", "n_neurons": 100, "activation": "relu"})
# layers_config.append({"layer_type": "fc", "n_neurons": 100, "activation": "relu"})
# layers_config.append({"layer_type": "fc", "n_neurons": 100, "activation": "relu"})
# layers_config.append({"layer_type": "softmax", "n_neurons": 2})

# layers_config.append({"layer_type": "input", "n_neurons": 2})
# layers_config.append({"layer_type": "fc", "n_neurons": 4})
# layers_config.append({"layer_type": "relu"})
# layers_config.append({"layer_type": "fc", "n_neurons": 4})
# layers_config.append({"layer_type": "relu"})
# layers_config.append({"layer_type": "regression", "n_neurons": 1})

layers_config.append({"layer_type": "input", "n_neurons": 2})
layers_config.append({"layer_type": "fc", "n_neurons": 4, "activation": 'relu'})
# layers_config.append({"layer_type": "fc", "n_neurons": 10, "activation": 'relu'})
layers_config.append({"layer_type": "logistic"})


from sklearn.datasets import load_iris

iris_dataset = load_iris()
X = iris_dataset["data"]
y = iris_dataset["target"]
X_processed = X[:, 2:]
Y_processed = (y == 0).astype('int').reshape(-1, 1)

nn = Net()
nn.bind_layers(layers_config)
nn.init_params(method="xavier")

Y_hat = nn.forward_prop(X_processed)
loss = nn.backward_prop(Y_processed)
Y_pred = (Y_hat > 0.5).astype("float")
print("acc:", np.sum(Y_pred == Y_processed) / Y_processed.shape[0])
print("loss:", loss)

costs = []
accs = []
for i in range(5000):
    Y_hat = nn.forward_prop(X_processed)
    loss = nn.backward_prop(Y_processed)
    
    Y_pred = (Y_hat > 0.5).astype("float")
    print(f"loss {i}: {np.round(loss, 3)}", end="\t")
    acc = np.round(np.sum(Y_pred == Y_processed) / Y_processed.shape[0], 3)
    print("acc:", acc)

    costs.append(loss)
    accs.append(acc)
    
    for layer in nn.layers:
        if hasattr(layer, "parameters"):
            layer.parameters["W"] = layer.parameters["W"] - 0.01 * layer.dJdW  
            layer.parameters["b"] = layer.parameters["b"] - 0.01 * layer.dJdb


import matplotlib.pyplot as plt

xx, yy = np.meshgrid(
                np.linspace(np.min(X_processed[:, 0]) - 1, np.max(X_processed[:, 0]) + 1, 500), 
                np.linspace(np.min(X_processed[:, 1]) - 1, np.max(X_processed[:, 1]) + 1, 500)
         )
X_new = np.c_[xx.ravel(), yy.ravel()]
y_proba = nn.forward_prop(X_new)
y_predict = (y_proba > 0.5).astype("float")
# zz1 = y_proba[:, 0].reshape(xx.shape)
zz = y_predict.reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# plt.contour(xx, yy, zz1)
axes[0].contourf(xx, yy, zz, cmap="jet")
axes[0].scatter(X_processed[:, 0], X_processed[:, 1], c=Y_processed)
axes[1].plot(costs, label="cost")
axes[1].plot(accs, label="acc")
axes[1].legend()
plt.show()
