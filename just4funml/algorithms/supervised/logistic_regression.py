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


class SoftmaxRegression:
    def __init__(self, learning_rate=0.1,
                 max_iter=100, regularized_type=None, alpha=None, early_stopping=False, 
                 random_state=None):  # C is regularization is applied, which level of regularization, early stopping option
        """
        learning_rate       : Learning rate of batch gradient descent
        max_iter            : Maximum iteration of gradient descent
        regularized_type    : [None, 'l2', 'l1'] Add penalty to the cost function of SoftmaxRegression, either 'l2 loss', 'l1 loss', or 'None'
        alpha               : Regularized hyperparameter (aka C), the lower alpha is, the more regularization is applied; alpha is ignored when regularized_type is None
        early_stopping      : If True, gradient descent will stop as soon as the cost reaches a minimum
        random_state        : Set seed to numpy random
        """ 

        if regularized_type not in (None, 'l1', 'l2'):
            raise ValueError('Unexpected regularized_type value (None, l1, l2)')

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.regularized_type = regularized_type
        self.alpha = alpha
        self.early_stopping = early_stopping
        np.random.seed(random_state)

    def fit(self, X, y):
        
        # Validate dimension of X and y
        if X.ndim != 2:
            raise ValueError('X is expected to be a 2d numpy array')
        elif y.ndim != 1:
            raise ValueError('y is expected to be a 1d numpy array')
        elif len(X) != len(y):
            raise ValueError('Number of rows of X must match with number of rows of y')

        # Add bias term to X
        X_with_bias = np.c_[np.ones((len(X), 1)), X]    # (m x (n + 1))    
        # One hot encode y
        Y_one_hot = preprocessing.to_one_hot(y)         # (m x k)

        self.K = Y_one_hot.shape[1]                     # Number of classes
        self.n = X_with_bias.shape[1] - 1               # Number of features
        m = len(X)                                      # Number of training examples

        # Initialize theta randomly
        self.Theta = np.random.randn(self.n + 1, self.K)    # ((n + 1) x k)
        
        # To keep track of lowest cost when early_stopping is enabled
        best_cost = np.inf  

        # Run batch gradient descent for max_iter times
        # Gradient descent can be stopped by:
        #   - max_iter reached
        #   - If early_stopping is enable, it stops when newly computed cost is higher than previous cost

        for epoch in range(self.max_iter):

            # Compute softmax score / logit for each class according to each training example 
            logits = X_with_bias @ self.Theta  # (m x k) = (m x (n + 1)) x ((n + 1) x k)

            # Apply softmax function to logits
            Y_proba_pred = self._softmax(logits)  # (m x k)

            error = Y_proba_pred - Y_one_hot  # (m x k)

            # Apply different regularization based to regularized_type to compute current gradients
            if self.regularized_type == 'l2':   # l2 loss
                gradients = (1 / m) * X_with_bias.T @ error + np.r_[np.zeros((1, self.K)), self.alpha * self.Theta[1:]]
            elif self.regularized_type == 'l1': # l1 loss
                raise NotImplementedError()
            else:   # Unregularized
                gradients = (1 / m) * X_with_bias.T @ error  # ((n + 1) x k) = ((m x (n + 1))^T) x (m x k)

            # Update Theta
            self.Theta -= self.learning_rate * gradients  # ((n + 1) x k)

            # If early_stopping is enable, stop iteration when the current cost is higher than previous cost
            if self.early_stopping:
                current_cost = self._compute_cost(X_with_bias, Y_one_hot)

                if current_cost < best_cost:
                    best_cost = current_cost
                else:
                    print(epoch, current_cost, 'early stopping!')
                    break

    def predict(self, X, return_in_prob=False):

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
        Compute cost of the model based on current theta and its respective cost function, cost function of SoftmaxRegression aka cross entropy
        """

        m = len(X_with_bias)
        logits = X_with_bias @ self.Theta
        Y_proba_pred = self._softmax(logits)

        # Add a tiny value epsilon to log(Y_proba_pred) to avoid getting nan values when Y_proba_pred contains any 0
        epsilon = 1e-7

        if self.regularized_type == 'l2':
            cross_entropy_loss = -np.mean(np.sum(Y_one_hot * np.log(Y_proba_pred + epsilon), axis=1))
            l2_loss = 1 / 2 * np.sum(np.square(self.Theta[1:]))
            cost = cross_entropy_loss + self.alpha * l2_loss

        elif self.regularized_type == 'l1':
            raise NotImplementedError()

        else:
            cost = - (1 / m) * np.sum(Y_one_hot * np.log(Y_proba_pred + epsilon))

        return cost

    def _softmax(self, logits):
        """
        Softmax function

        - Apply exponent to every logit and divide every logit by its sum of exponent across every classes for each example
        - Take (m x k) return (m x k)
        """

        exps_logits = np.exp(logits)  # (m x k)
        sums_exp_logits_across_classes = np.sum(exps_logits, axis=1, keepdims=True)  # (m x 1)

        return exps_logits / sums_exp_logits_across_classes  # (m x k) = (m x k) / (m x 1)


