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

import just4funml.utils.preprocessing as preprocessing


class LogisticRegression:
    def __init__(self, learning_rate=0.1, max_iter=100, regularized_type=None, alpha=None, random_state=None):

        if regularized_type not in (None, 'l1', 'l2'):
            raise ValueError('Unexpected regularized_type value (None, l1, l2)')

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.regularized_type = regularized_type
        self.alpha = alpha
        np.random.seed(random_state)

    def fit(self, X, y):
        X_with_bias = np.c_[np.ones((len(X), 1)), X]
        y = y.astype(np.float64).reshape(len(y), 1)    # Assume y is boolean numpy array

        m = len(X)
        n = X.shape[1]

        self.theta = np.random.randn(n + 1, 1)

        for epoch in range(self.max_iter):
            y_proba = self._sigmoid(X_with_bias @ self.theta)
            errors = y - y_proba
            gradient = (- 1 / m) * X_with_bias.T @ errors

            if self.regularized_type == 'l2':
                gradient += (self.alpha / m) * np.r_[0, self.theta[1:]]
            elif self.regularized_type == 'l1':
                gradient += (self.alpha / m) * np.r_[0, np.sign(self.theta[1:])]

            self.theta -= self.learning_rate * gradient


    def predict(self, X, return_in_prob=False):
        X_with_bias = np.c_[np.ones((len(X), 1)), X]
        y_proba = self._sigmoid(X_with_bias @ self.theta)
        y_proba = np.c_[y_proba, 1 - y_proba]

        if not return_in_prob:
            y_pred = y_proba[:, 0] >= 0.5
            return y_pred

        return y_proba
        
    def _compute_cost(self, X_with_bias, y_actual):

        m = len(X_with_bias)

        # y_actual is boolean numpy array
        y_actual = y_actual.astype(np.float64).reshape(len(y_actual), 1)
        y_proba = self._sigmoid(X_with_bias @ self.theta)
        
        cost = (-1 / m) * np.sum((y_actual * np.log(y_proba)) - ((1 - y_actual) * np.log(1 - y_proba)))

        if self.regularized_type == 'l2':
            cost += (self.alpha / (2 * m)) * np.sum(np.square(self.theta[1:])) 
        elif self.regularized_type == 'l1':
            cost += (self.alpha / m) * np.sum(np.abs(self.theta[1:]))

        return cost 

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


class SoftmaxRegression:
    def __init__(self, learning_rate=0.1,
                 max_iter=100, regularized_type=None, alpha=None, random_state=None):
        """
        learning_rate       : Learning rate of batch gradient descent
        max_iter            : Maximum iteration of gradient descent
        regularized_type    : [None, 'l2', 'l1'] Add penalty to the cost function of SoftmaxRegression, either 'l2 loss', 'l1 loss', or 'None'
        alpha               : Regularized hyperparameter (aka C), the lower alpha is, the more regularization is applied; alpha is ignored when regularized_type is None
        random_state        : Set seed to numpy random
        """ 

        if regularized_type not in (None, 'l1', 'l2'):
            raise ValueError('Unexpected regularized_type value (None, l1, l2)')

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.regularized_type = regularized_type
        self.alpha = alpha
        np.random.seed(random_state)

    def fit(self, X, y):
        """
        Fit the model with batch gradient descent
        """

        # Validate dimension of X and y
        if X.ndim != 2:
            raise ValueError('X is expected to be a 2d numpy array')
        elif y.ndim != 1:
            raise ValueError('y is expected to be a 1d numpy array')
        elif len(X) != len(y):
            raise ValueError('Number of rows of X must match with number of rows of y')

        # Prepare X and Y
        X_with_bias = np.c_[np.ones((len(X), 1)), X]    # (m x (n + 1))    
        Y_one_hot = preprocessing.to_one_hot(y)         # (m x k)

        # Declare some useful variables
        m = len(X)                                      # Number of training examples
        self.K = Y_one_hot.shape[1]                     # Number of classes
        self.n = X_with_bias.shape[1] - 1               # Number of features
        self.Theta = np.random.randn(self.n + 1, self.K)    # ((n + 1) x k)
     
        # Run batch gradient descent for max_iter times
        # Gradient descent can be stopped by:
        #   - max_iter reached
        for epoch in range(self.max_iter):

            # Compute logits and apply softmax function
            logits = X_with_bias @ self.Theta           # (m x k) = (m x (n + 1)) x ((n + 1) x k)
            Y_proba_pred = self._softmax(logits)        # (m x k)
            error = Y_one_hot - Y_proba_pred            # (m x k)

            if epoch % 500 == 0:
                print(epoch, self._compute_cost(X_with_bias, Y_one_hot))

            # Compute gradient of cost function
            gradients = (-1 / m) * X_with_bias @ error  # ((n + 1) x k) = ((m x (n + 1))^T) x (m x k)

            # Apply regularization if any
            if self.regularized_type == 'l2':   # l2 loss
                gradients += self.alpha * np.r_[np.zeros((1, self.K)), self.Theta[1:]]

            elif self.regularized_type == 'l1': # l1 loss
                gradients += self.alpha * np.r_[np.zeros((1, self.K)), np.sign(self.Theta[1:])]
            
            # Update Theta
            self.Theta -= self.learning_rate * gradients  # ((n + 1) x k)


    def fit_with_early_stopping(self, X_train, y_train, X_valid, y_valid):
        """
        If fit the model with early stopping, batch gradient descent will stop as soon as 
        the cost of validation set is higher than the previous cost of it
        """

        # Validate dimension of X and y
        if X_train.ndim != 2 or X_valid.ndim != 2:
            raise ValueError('X is expected to be a 2d numpy array')
        elif y_train.ndim != 1 or y_valid.ndim != 1:
            raise ValueError('y is expected to be a 1d numpy array')
        elif len(X_train) != len(y_train) or len(X_valid) != len(y_valid):
            raise ValueError('Number of rows of X must match with number of rows of y')

        # Add bias term to X
        X_train_w_bias = np.c_[np.ones((len(X_train), 1)), X_train]
        X_valid_w_bias = np.c_[np.ones((len(X_valid), 1)), X_valid]

        # One hot encode y
        Y_train_one_hot = preprocessing.to_one_hot(y_train)
        Y_valid_one_hot = preprocessing.to_one_hot(y_valid)

        # Declare some useful variables
        self.K = Y_train_one_hot.shape[1]                  # Number of classes
        self.n = X_train_w_bias.shape[1] - 1               # Number of features
        m = len(X_train)                                   # Number of training examples

        # Initialize theta
        self.Theta = np.random.randn(self.n + 1, self.K)

        # Keep track of lowest cost of validation set
        best_valid_cost = np.inf

        for epoch in range(self.max_iter):
            
            # Compute Y_proba_pred
            logits = X_train_w_bias @ self.Theta  # (m x k) = (m x (n + 1)) x ((n + 1) x k)
            Y_proba_pred = self._softmax(logits)  # (m x k)
            
            # Compute error term
            error = Y_train_one_hot - Y_proba_pred  # (m x k)

            # Compute gradients of cost function
            gradients = (-1 / m) * X_train_w_bias @ error  # ((n + 1) x k) = ((m x (n + 1))^T) x (m x k)

            # Apply regularization if any
            if self.regularized_type == 'l2':   # l2 loss
                gradients += self.alpha * np.r_[np.zeros((1, self.K)), self.Theta[1:]]

            elif self.regularized_type == 'l1': # l1 loss
                gradients += self.alpha * np.r_[np.zeros((1, self.K)), np.sign(self.Theta[1:])]

            # Update Theta
            self.Theta -= self.learning_rate * gradients  # ((n + 1) x k)

            # Stop iteration when the current cost of validation set is higher than previous cost
            current_cost = self._compute_cost(X_valid_w_bias, Y_valid_one_hot)

            if epoch % 500 == 0:
                print(epoch, current_cost)

            if current_cost < best_valid_cost:
                best_valid_cost = current_cost
            else:
                print(epoch, current_cost, 'early stopping!')
                break

    def predict(self, X, return_in_prob=False):
        """
        Make prediction with input X using current available Theta

        - Add bias to X
        - Compute logits with current Theta
        - Apply softmax function to logits to get the probabilities
        - Either Return the Y_proba directly or get the class with highest prob then return
        """
        if X.shape[1] != self.n:
            raise ValueError(f'Number of columns of X must equal to {self.n}')

        X_with_bias = np.c_[np.ones((len(X), 1)), X]

        # Compute score s_k(X) for each class k for each example
        logits = X_with_bias @ self.Theta

        # estimate probability of each class by applying softmax function
        Y_proba_pred = self._softmax(logits)

        if not return_in_prob:
            # The class with highest probability is the predicted class
            y_pred = np.argmax(Y_proba_pred, axis=1)
            return y_pred

        return Y_proba_pred

    def _compute_cost(self, X_with_bias, Y_one_hot):
        """
        Compute cost of the model based on current theta and its respective cost function (cross entropy loss)
        """
        
        logits = X_with_bias @ self.Theta
        Y_proba_pred = self._softmax(logits)

        # Add a tiny value epsilon to log(Y_proba_pred) to avoid getting nan values when Y_proba_pred contains any 0
        epsilon = 1e-7

        cross_entropy_loss = -np.mean(np.sum(Y_one_hot * np.log(Y_proba_pred + epsilon), axis=1))

        if self.regularized_type == 'l2':
            l2_loss = 1 / 2 * np.sum(np.square(self.Theta[1:]))
            cross_entropy_loss += self.alpha * l2_loss

        elif self.regularized_type == 'l1':
            l1_loss = np.sum(np.abs(self.Theta[1:]))
            cross_entropy_loss += self.alpha * l1_loss

        return cross_entropy_loss

    def _softmax(self, logits):
        """
        Softmax function

        - Apply exponent to every logit and divide every logit by its sum of exponent across every classes for each example
        - Take (m x k) return (m x k)
        """

        exps_logits = np.exp(logits)  # (m x k)
        sums_exp_logits_across_classes = np.sum(exps_logits, axis=1, keepdims=True)  # (m x 1)

        return exps_logits / sums_exp_logits_across_classes  # (m x k) = (m x k) / (m x 1)
