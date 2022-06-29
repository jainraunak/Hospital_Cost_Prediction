import numpy as np

class LinearRegression:
    def __init__(self):
        self.weight = None

    def fit(self,X,Y):
        """
        Learn weights for features using Linear Regression
        :param X: Training data
        :param Y: Training labels
        :return: Nothing
        """
        A = np.transpose(X)
        B = np.matmul(A, X)
        C = np.matmul(A, Y)
        D = np.linalg.inv(B)
        self.weight = np.matmul(D, C)

    def predict(self,X):
        """
        Prediction using linear regression
        :param X: Test data
        :return: Predicted values for test data
        """
        ans = np.matmul(X, self.weight)
        return ans