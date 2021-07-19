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


class LinearRegression:

    def __init__(self, features, labels):
        if len(features) == len(labels):
            self.features = np.array(features).reshape((len(features), 1))
            self.labels = np.array(labels).reshape((len(labels), 1))
            self.m = np.size(self.features, 0)
            self.n = np.size(self.features, 1)
            self.features = np.concatenate((np.ones((self.m, 1)), self.features), axis=1)  # m x (n + 1)
            self.theta = np.empty((self.n + 1, 1))  # (n + 1) x 1
            self.predictions = np.empty((self.m, 1))
            self.hypothesis()
            print(train_cv_test_split)
            train_cv_test_split(features)
        else:
            raise Exception('Number of features is not same with that of label')

    def hypothesis(self):
        self.predictions = np.matmul(self.features, self.theta)
        return self.predictions

    def cost_function(self):
        j = (1 / (2 * self.m)) * np.sum(np.square(np.subtract(self.predictions, self.labels)))
        return j

