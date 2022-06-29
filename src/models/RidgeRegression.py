import numpy as np


class RidgeRegression:
    def __init__(self, penalty=0.001, index_bias=-1):
        self.weight = None
        self.penalty = penalty  # Ridge Regression Penalty
        self.index_bias = index_bias  # Index of bias column

    def fit(self, X, Y):
        """
        Learn weights for features using ridge regression
        :param X: Training data
        :param Y: Training labels
        :return: Nothing
        """
        A = np.transpose(X)
        B = np.matmul(A, X)
        C = np.matmul(A, Y)
        nr = X.shape[1]
        Q = np.zeros((nr, nr))
        np.fill_diagonal(Q, self.penalty)
        Q[self.index_bias][self.index_bias] = 0  # Don't penalise bias
        D = np.add(B, Q)
        P = np.linalg.inv(D)
        self.weight = np.matmul(P, C)

    def predict(self, X):
        """
        Prediction using ridge regression
        :param X: Test data
        :return: Predicted values for test data
        """
        ans = np.matmul(X, self.weight)
        return ans
