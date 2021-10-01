import numpy as np
from net import Net 
from sklearn.datasets import load_iris

# Load iris dataset
iris_dataset = load_iris()
X = iris_dataset["data"][:, 2:]
y = iris_dataset["target"]
y = (y == 0).astype('int').reshape(-1, 1)

# My own neural network :)
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
layers_config.append({"layer_type": "fc", "n_neurons": 4, "activation": 'sigmoid'})
# layers_config.append({"layer_type": "fc", "n_neurons": 10, "activation": 'relu'})
layers_config.append({"layer_type": "logistic"})

nn = Net()
nn.bind_layers(layers_config)
check_params = nn.init_params(method="xavier", seed=42)
check_X = X[:4]
check_y = y[:4]

# Manual check for forward propagation
y_proba = nn.forward_prop(check_X)

Z1 = np.dot(check_X, check_params["W1"]) + check_params["b1"]
A1 = 1 / (1 + np.exp(-Z1))
Z2 = np.dot(A1, check_params["W2"]) + check_params["b2"]
A2 = 1 / (1 + np.exp(-Z2))
# print(y_proba)
# print(A2)

# Manual check for backward propagation
print("Backprop model")
cost = nn.backward_prop(check_y)

print("\nManual Backprop")
loss_manual = - (check_y * np.log(A2) + (1- check_y) * np.log(1 - A2))
cost_manual = np.squeeze(np.mean(loss_manual, axis=0))
print(f"loss: {loss_manual}")
dLdA2 = (- check_y / A2) + ((1 - check_y) / (1 - A2))
print(f"dLdA2: {dLdA2}")
dLdZ2 = dLdA2 * (1 - A2) * A2
print(f"dLdZ2: {dLdZ2}")
dJdW2 = 1 / 4 * np.dot(A1.T, dLdZ2)
print(f"dLdW2: {dJdW2}")
dJdb2 = 1 / 4 * np.sum(dLdZ2, axis=0, keepdims=True)
print(f"dJdb2: {dJdb2}")
dLdA1 = np.dot(dLdZ2, check_params["W2"].T)
print(f"dLdA1: {dLdA1}")

dLdZ1 = dLdA1 * (1 - A1) * A1
print(f"dLdZ1: {dLdZ1}")
dJdW1 = 1 / 4 * np.dot(check_X.T, dLdZ1)
print(f"dJdW1: {dJdW1}")
dJdb1 = 1 / 4 * np.sum(dLdZ1, axis=0, keepdims=True)
print(f"dJdb1: {dJdb1}")
dLdA0 = np.dot(dLdZ1, check_params["W1"].T) 
print(f"dLdA0: {dLdA0}")