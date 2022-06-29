from src.models.RidgeRegression import RidgeRegression
from src.CrossValidation import KFoldCrossValidation
import numpy as np
import os
import pandas as pd

def solve_b(X_train,Y_train,X_test,results_path,regc,k):
    """
    Do Part b i.e. use ridge regression to predict total costs.
    :param X_train: Training data
    :param Y_train: Training labels
    :param X_test: Test data
    :param results_path: Path to save results
    :param regc: List storing the regression penalties which will be used in ridge regression
    :param k: k of k-fold CV
    :return: Nothing
    """

    bp = regc[0]        # Best Regression Penalty
    best_r2_score = 0   # Best k-fold R2 Score

    df_result = pd.DataFrame(columns=['Ridge Penalty',str(k)+' - fold R2 Score'])
    for l in regc:
        model = RidgeRegression(penalty=l,index_bias=-1)
        score = KFoldCrossValidation(X_train=X_train,Y_train=Y_train,model=model,k=k)

        dic = {'Ridge Penalty':l, str(k)+' - fold R2 Score': score }
        df_result = df_result.append(dic,ignore_index=True)

        if score > best_r2_score:
            best_r2_score = score
            bp = l

    model = RidgeRegression(penalty=bp)     # Best Ridge Regression model
    model.fit(X=X_train,Y=Y_train)
    predictions = model.predict(X=X_test)
    weights = model.weight                  # Weights of features

    results_path = os.path.join(results_path, "Part_b")
    try:
        os.mkdir(results_path)
    except:
        do_nothing = True

    np.savetxt(os.path.join(results_path, "weights.txt"), weights)
    np.savetxt(os.path.join(results_path, "predictions.txt"), predictions)
    df_result.to_csv(os.path.join(results_path, "results_ridge_penalty.csv"),index=False)