import tensorflow as tf
import tensorflow.keras as keras
from sklearn.datasets import load_iris
import numpy as np


iris_dataset = load_iris()
X = iris_dataset["data"]
y = iris_dataset["target"]
X_processed = X[:, 2:]
Y_processed = np.zeros((X.shape[0], 2))
Y_processed[y == 0, 0] = 1
Y_processed[y != 0, 1] = 1

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(2)),
    keras.layers.Dense(units=4, activation="relu"),
    keras.layers.Dense(units=1, activation="sigmoid")
])
model.compile(optimizer="SGD", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(x=X_processed, y=Y_processed, epochs=100)