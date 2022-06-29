import os

import numpy as np

from src.models.LinearRegression import LinearRegression


def solve_a(X_train, Y_train, X_test, results_path):
    """
    Do Part a i.e. use linear regression to predict total costs.
    :param X_train: Training data
    :param Y_train: Training labels
    :param X_test: Test data
    :param results_path: Path to store the results
    :return: Nothing
    """

    model = LinearRegression()
    model.fit(X=X_train, Y=Y_train)
    prediction = model.predict(X=X_test)
    weights = model.weight  # weights of the features

    results_path = os.path.join(results_path, "Part_a")
    try:
        os.mkdir(results_path)
    except:
        do_nothing = True

    np.savetxt(os.path.join(results_path, "weights.txt"), weights)
    np.savetxt(os.path.join(results_path, 'predictions.txt'), prediction)
