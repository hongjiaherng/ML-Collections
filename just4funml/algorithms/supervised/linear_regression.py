import sys
import os
import numpy as np

if __name__ == '__main__':
    import re
    current_folder = re.split(r'[\\/]', os.path.abspath(os.getcwd()))[-1]

    if current_folder != 'supervised':
        print("Please make sure you are in 'supervised/' folder when executing this file directly!")
        sys.exit(0)
    else:
        module_path = os.path.abspath(os.path.join('../../..'))
        if module_path not in sys.path:
            sys.path.append(module_path)

from just4funml.utils.helper import train_cv_test_split

class Regression:
    def __init__(self, max_iter=100, learning_rate=0.1):
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def fit(self, X, y):
        m, n = X.shape
        X_with_bias = np.c_[np.ones((m, 1)), X]

        self.theta = np.random.randn(n + 1, 1)

        for i in range(self.max_iter):
            y_pred = X_with_bias @ self.theta
            errors = y_pred - y
            gradients = (1 / m) * X_with_bias.T @ errors
            self.theta -= self.learning_rate * gradients

    def predict(self, X):
        m = X.shape[0]
        X_with_bias = np.c_[np.ones((m, 1)), X]
        y_pred = X_with_bias @ self.theta
        return y_pred
        


class LinearRegression(Regression):
    def __init__(self, max_iter=100, learning_rate=0.1, normal_eq=False):
        self.normal_eq = normal_eq
        super().__init__(max_iter=max_iter, learning_rate=learning_rate)

    def fit(self, X, y):
        if self.normal_eq:
            X_with_bias = np.c_[np.ones((len(X), 1)), X]
            self.theta = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        else:
            super().fit(X, y)

    def predict(self, X):
        return super().predict(X)

class RidgeRegression(Regression):
    def __init__(self, max_iter=100, learning_rate=0.1, normal_eq=False, alpha=0.1):
        self.normal_eq = normal_eq
        self.alpha = alpha
        super().__init__(max_iter=max_iter, learning_rate=learning_rate)
    def fit(self, X, y):
        if self.normal_eq:
            pass
        else:
            m, n = X.shape
            X_with_bias = np.c_[np.ones((m, 1)), X]

            self.theta = np.random.randn(n + 1, 1)

            for i in range(self.max_iter):
                


class LassoRegression(Regression):
    pass

class ElasticNet(Regression):
    pass

# class LinearRegression:

#     def __init__(self, features, labels):
#         if len(features) == len(labels):
#             self.features = np.array(features).reshape((len(features), 1))
#             self.labels = np.array(labels).reshape((len(labels), 1))
#             self.m = np.size(self.features, 0)
#             self.n = np.size(self.features, 1)
#             self.features = np.concatenate((np.ones((self.m, 1)), self.features), axis=1)  # m x (n + 1)
#             self.theta = np.empty((self.n + 1, 1))  # (n + 1) x 1
#             self.predictions = np.empty((self.m, 1))
#             self.hypothesis()
#             print(train_cv_test_split)
#             train_cv_test_split(features)
#         else:
#             raise Exception('Number of features is not same with that of label')

#     def hypothesis(self):
#         self.predictions = np.matmul(self.features, self.theta)
#         return self.predictions

#     def cost_function(self):
#         j = (1 / (2 * self.m)) * np.sum(np.square(np.subtract(self.predictions, self.labels)))
#         return j
