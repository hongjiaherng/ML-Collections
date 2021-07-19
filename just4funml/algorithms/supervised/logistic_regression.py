import numpy as np
import copy


class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, tolerance=1e-07,
                 max_iter=100, regularized_type=None, alpha=0.1, early_stopping=False):  # C is regularization is applied, which level of regularization, early stopping option
        """
        :param learning_rate: Learning rate
        :param tolerance: Tolerance rate. Gradient descent stop by either max_iter reach or the norm of gradient vector is less than toleratnce rate
        :param max_iter: Maximum iteration of gradient descent
        """
        if regularized_type not in (None, 'l1', 'l2'):
            raise ValueError('Unexpected regularized_type value (None, l1, l2)')

        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.regularized_type = regularized_type
        self.alpha = alpha
        self.early_stopping = early_stopping

    def fit(self, X, y):

        if X.ndim != 2:
            raise ValueError('X is expected to be a 2d numpy array')
        elif y.ndim != 1:
            raise ValueError('y is expected to be a 1d numpy array')
        elif len(X) != len(y):
            raise ValueError('Number of rows of X must match with number of rows of y')

        X_with_bias = np.c_[np.ones((len(X), 1)), X]  # (m x (n + 1))
        Y_one_hot = self._to_one_hot(y)  # (m x k)

        self.K = Y_one_hot.shape[1]
        m = len(X)
        self.n = X_with_bias.shape[1] - 1
        self.Theta = np.random.randn(self.n + 1, self.K)  # ((n + 1) x k)

        # TODO: Look here
        best_cost = np.infty
        best_epoch = None
        best_Theta = None
        # TODO: Look here

        for epoch in range(self.max_iter):
            logits = X_with_bias @ self.Theta  # (m x k) = (m x (n + 1)) x ((n + 1) x k)
            Y_proba_pred = self._softmax(logits)  # (m x k)

            error = Y_proba_pred - Y_one_hot  # (m x k)

            if self.regularized_type == 'l2':
                gradients = (1 / m) * X_with_bias.T @ error + np.r_[np.zeros((1, self.K)), self.alpha * self.Theta[1:]]
            elif self.regularized_type == 'l1':
                raise NotImplementedError()
            else:
                gradients = (1 / m) * X_with_bias.T @ error  # ((n + 1) x k) = ((m x (n + 1))^T) x (m x k)

            if np.linalg.norm(gradients) < self.tolerance:
                print('here')
                break

            self.Theta -= self.learning_rate * gradients  # ((n + 1) x k)

        # TODO: Look here
            if self.early_stopping:
                cost = self._compute_cost(X_with_bias, Y_one_hot)

                if cost < best_cost:
                    best_cost = cost
                    best_epoch = epoch
                    best_Theta = self.Theta

        if self.early_stopping:
            self.Theta = best_Theta
        # TODO: Look here

    def predict(self, X, return_in_prob=False):
        # Compute score s_k(X) for each class k
        # estimate probability of each class by applying softmax function
        # the class with highest probability is the predicted class
        if X.shape[1] != self.n:
            raise ValueError(f'Number of columns of X must equal to {self.n}')

        X_with_bias = np.c_[np.ones((len(X), 1)), X]
        logits = X_with_bias @ self.Theta
        Y_proba_pred = self._softmax(logits)

        if not return_in_prob:
            y_pred = np.argmax(Y_proba_pred, axis=1)
            return y_pred
        return Y_proba_pred

    def _compute_cost(self, X_with_bias, Y_one_hot):

        m = len(X_with_bias)
        logits = X_with_bias @ self.Theta
        Y_proba_pred = self._softmax(logits)

        # Add a tiny value epsilon to log(Y_proba_pred) to avoid getting nan values
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

        exps_logits = np.exp(logits)  # (m x k)
        sums_exp_logits_across_classes = np.sum(exps_logits, axis=1, keepdims=True)  # (m x 1)

        return exps_logits / sums_exp_logits_across_classes  # (m x k) = (m x k) / (m x 1)

    def _to_one_hot(self, y):

        m = len(y)
        unique_classes = np.unique(y)
        Y_one_hot = np.zeros(shape=(m, unique_classes.size))

        for i in range(m):
            selected_col_index = np.argwhere(unique_classes == y[i])[0]
            Y_one_hot[i, selected_col_index] = 1.0

        return Y_one_hot



from sklearn import datasets
np.random.seed(2042)

iris = datasets.load_iris()

X = iris['data'][:, (2, 3)]
y = iris['target']

test_ratio = 0.2
validation_ratio = 0.2
total_size = len(X)

test_size = int(total_size * test_ratio)
validation_size = int(total_size * validation_ratio)
train_size = total_size - test_size - validation_size

rnd_indices = np.random.permutation(total_size)

X_train = X[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]
X_valid = X[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]
X_test = X[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]

softmax_reg = SoftmaxRegression(learning_rate=0.1, tolerance=-np.inf, max_iter=5001, regularized_type='l2', alpha=0.1)
softmax_reg.fit(X_train, y_train)
# print(softmax_reg.Theta)
y_val_pred = softmax_reg.predict(X_valid)
print(np.mean(y_val_pred == y_valid))


