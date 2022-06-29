from sklearn.linear_model import LassoLars


class LassoRegression:
    def __init__(self, penalty=0.001):
        self.weight = None
        self.penalty = penalty  # Lasso Regression Penalty
        self.model = None

    def fit(self, X, Y):
        """
        Learn weights for features using lasso regression
        :param X: Training data
        :param Y: Training labels
        :return: Nothing
        """
        self.model = LassoLars(alpha=self.penalty)
        self.model.fit(X, Y)  # Fit lasso regression model
        self.weight = self.model.coef_

    def predict(self, X):
        """
        Prediction using lasso regression
        :param X: Test data
        :return: Predicted values for test data
        """
        ans = self.model.predict(X)
        return ans
