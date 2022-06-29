import numpy as np
from sklearn.metrics import r2_score


def KFoldCrossValidation(X_train, Y_train, model, k=10):
    """
    Do k-fold cross validation using the model. The score function of k-fold CV is R2 Score.
    :param X_train: Training data
    :param Y_train: Test data
    :param model: model to be used for learning weights of features
    :param k: k of k-fold cross CV
    :return: k-fold CV R2 Score
    """
    r = int(X_train.shape[0] / k)
    k_fold_r2_score = 0.0
    fold = 1
    while (fold <= k):
        X1 = X_train[0:(fold - 1) * r]
        Y1 = Y_train[0:(fold - 1) * r]
        X2 = X_train[(fold - 1) * r:fold * r]
        Y2 = Y_train[(fold - 1) * r:fold * r]
        X3 = X_train[fold * r:(fold + 1) * r]
        Y3 = Y_train[fold * r:(fold + 1) * r]

        X_tra = np.concatenate((X1, X3), axis=0)
        Y_tra = np.concatenate((Y1, Y3), axis=0)

        model.fit(X=X_tra, Y=Y_tra)
        predictions = model.predict(X2)
        k_fold_r2_score += r2_score(Y2, predictions)
        fold += 1

    k_fold_r2_score /= k
    return k_fold_r2_score
