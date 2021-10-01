import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_circles
from sklearn.linear_model import LogisticRegression
from net import Net
import tensorflow.keras as keras
import sys

# Load circles
X, y = make_circles(n_samples=1000, noise=0.03, random_state=42)
y = y.reshape(-1, 1)

# Load iris dataset
# iris_dataset = load_iris()
# X = iris_dataset["data"][:, 2:]
# y = iris_dataset["target"]
# y = (y == 0).astype('int').reshape(-1, 1)

# Learning algorithms
# Logistic regression
log_reg = LogisticRegression()
log_reg.fit(X, y[:, 0])

# TensorFlow Keras NN
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(2)),
    keras.layers.Dense(units=4, activation="tanh"),
    keras.layers.Dense(units=4, activation="tanh"),
    keras.layers.Dense(units=4, activation="tanh"),
    keras.layers.Dense(units=1, activation="sigmoid")
])
model.compile(optimizer="SGD", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(x=X, y=y, epochs=1000)

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
layers_config.append({"layer_type": "fc", "n_neurons": 4, "activation": 'tanh'})
layers_config.append({"layer_type": "fc", "n_neurons": 4, "activation": 'tanh'})
layers_config.append({"layer_type": "fc", "n_neurons": 4, "activation": 'tanh'})
# layers_config.append({"layer_type": "fc", "n_neurons": 10, "activation": 'relu'})
layers_config.append({"layer_type": "logistic"})

nn = Net()
nn.bind_layers(layers_config)
nn.init_params(method="xavier")

costs = []
accs = []
for i in range(2000):
    y_proba = nn.forward_prop(X)
    loss = nn.backward_prop(y)
    
    y_pred = (y_proba > 0.5).astype("float")
    acc = np.round(np.sum(y_pred == y) / y.shape[0], 3)

    print(f"loss {i}: {np.round(loss, 3)}", end="\t")
    print("acc:", acc)

    costs.append(loss)
    accs.append(acc)
    
    for layer in nn.layers:
        if hasattr(layer, "parameters"):
            layer.parameters["W"] = layer.parameters["W"] - 0.08 * layer.dJdW  
            layer.parameters["b"] = layer.parameters["b"] - 0.08 * layer.dJdb


def plot_result(predictors):
    # Plotting
    xx, yy = np.meshgrid(
        np.linspace(np.min(X[:, 0]) - 0.5, np.max(X[:, 0]) + 0.5, 500),
        np.linspace(np.min(X[:, 1]) - 0.5, np.max(X[:, 1]) + 0.5, 500),
    )
    X_mesh = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, predictor in zip(axes, predictors):
        if predictor.__class__.__name__ == "Net":
            y_proba_mesh = predictor.forward_prop(X_mesh)
            y_pred_mesh = (y_proba_mesh > 0.5).astype("float")
        else:
            y_proba_mesh = predictor.predict(X_mesh)
            y_pred_mesh = (y_proba_mesh > 0.5).astype("float")
        
        zz = y_pred_mesh.reshape(xx.shape)
        ax.contourf(xx, yy, zz, cmap="tab10")
        ax.scatter(X[:, 0], X[:, 1], c=y, s=1)

    plt.show()

plot_result([model, nn])
