from src.models.RidgeRegression import RidgeRegression
from src.models.LassoRegression import LassoRegression
from src.CrossValidation import KFoldCrossValidation
from src.feature_engineering import feature_engineering
from termcolor import colored
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random

def plots(predictions,actual,results_path,LOS):
    """
    Make plots
    :param predictions: Predicted values
    :param actual: Actual values
    :param results_path: Path to store the results
    :param LOS: Length of Stay training data
    :return: Nothing
    """

    residuals = predictions-actual

    results_path = os.path.join(results_path,'Plots')
    try:
        os.mkdir(results_path)
    except:
        do_nothing = True

    # Create Total Costs vs LOS plot
    plt.scatter(LOS,actual,s=1)
    plt.title('Total Costs vs Length of Stay')
    plt.xlabel('Length of Stay')
    plt.ylabel('Total Costs')
    plt.savefig(os.path.join(results_path,'TotalCostsVsLOS.png'))
    plt.clf()

    # Create Predicted Total Costs vs Actual Total Costs plot
    val = np.polyfit(x=actual,y=predictions,deg=1)
    m = val[0]
    b = val[1]
    plt.scatter(actual,predictions,s=1)
    plt.plot(actual,m*actual+b,c='k')   # Best fit line
    plt.title('Predictions vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predictions')
    plt.savefig(os.path.join(results_path,'PredictionsVsActual.png'))
    plt.clf()

    # Create residuals vs Actual Total Costs plot
    plt.scatter(actual,residuals,s=1)
    plt.title('Residuals vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Residuals')
    plt.savefig(os.path.join(results_path,'ResidualsVsActual.png'))
    plt.clf()

    # Create histogram plot of residuals
    plt.hist(residuals)
    plt.title('Density Plot of Residuals')
    plt.ylabel('Count')
    plt.xlabel('Residuals')
    plt.savefig(os.path.join(results_path,'Histogram.png'))
    plt.clf()

def solve_c(train_data,test_data,results_path,k,get_features_importance,reg_lower_limit,reg_upper_limit,random_searches):
    """
    Do part c i.e. use feature engineering to improve model performance.
    :param train_data: Training data
    :param test_data: Test data
    :param results_path: Path to store the results
    :param k: k of k-fold CV
    :param get_important_features: Whether to get important features or not using Lasso Regression
    :param reg_lower_limit: Lower limit of regularisation penalty
    :param reg_upper_limit: Upper limit of regularisation penalty
    :param random_searches: Number of random searches in [reg_lower_limit,reg_upper_limit] to find the
                            best regularisation penalty
    :return: Nothing
    """

    X_train = train_data.iloc[:,1:-1]
    Y_train = np.asarray(train_data['Total Costs'].values)
    X_test = test_data.iloc[:,1:-1]

    col = np.ones(X_train.shape[0])
    X_train['bias'] = col
    col = np.ones(X_test.shape[0])
    X_test['bias'] = col

    model = RidgeRegression(penalty=0.001)
    without_feature_engineering_k_fold_r2_score = KFoldCrossValidation(X_train=np.asarray(X_train),
                                                                       Y_train=Y_train,model=model,k=k)

    print(colored(str(k) + '-Fold R2 Score (Without Feature Engineering) : ' +
                  str(without_feature_engineering_k_fold_r2_score),'cyan'))

    # Remove outliers
    non_outliers = (Y_train <= 2e5)
    Y_train = Y_train[non_outliers]
    X_train = X_train.loc[non_outliers]

    data = pd.concat([X_train, X_test], axis=0)

    # Feature Engineering
    data = feature_engineering(data=data)

    index_bias = list(data.columns).index('bias')       # Index of bias in features list

    X_train = np.asarray(data.iloc[0:X_train.shape[0]])
    X_test = np.asarray(data.iloc[X_train.shape[0]:])

    print(colored('Finished Feature Engineering','cyan'))

    best_r2_score = 0       # Best k-fold R2 score
    bp = None               # Best Ridge Regression Penalty
    df_result = pd.DataFrame(columns=['Ridge Penalty',str(k)+' - fold R2 Score'])
    for idx in range(random_searches):
        l = random.uniform(reg_lower_limit,reg_upper_limit)
        model = RidgeRegression(penalty=l,index_bias=index_bias)
        score = KFoldCrossValidation(X_train=X_train, Y_train=Y_train, model=model, k=k)

        dic = {'Ridge Penalty': l, str(k) + ' - fold R2 Score': score}
        df_result = df_result.append(dic, ignore_index=True)

        if score > best_r2_score:
            best_r2_score = score
            bp = l

    print(colored('Best '+str(k)+'-Fold R2 Score after feature engineering : '+str(best_r2_score),'cyan'))

    improvemet_r2_score = (best_r2_score-without_feature_engineering_k_fold_r2_score)*100
    improvemet_r2_score /= without_feature_engineering_k_fold_r2_score
    improvemet_r2_score = round(improvemet_r2_score,2)

    print(colored('Improvement in '+str(k)+' - Fold R2 Score due to feature engineering : '
                  +str(improvemet_r2_score)+' %','cyan'))

    model = RidgeRegression(penalty=bp,index_bias=index_bias)
    model.fit(X=X_train,Y=Y_train)
    predictions = model.predict(X=X_test)
    weights = model.weight

    results_path = os.path.join(results_path, "Part_c")
    try:
        os.mkdir(results_path)
    except:
        do_nothing = True

    np.savetxt(os.path.join(results_path, "weights.txt"), weights)
    np.savetxt(os.path.join(results_path, "predictions.txt"), predictions)
    df_result.to_csv(os.path.join(results_path, "results_ridge_penalty.csv"),index=False)

    train_predictions = model.predict(X_train)

    plots(predictions=train_predictions,actual=Y_train,
          results_path=results_path,
          LOS=data['Length of Stay'][0:X_train.shape[0]].values)

    if get_features_importance:
        # Use Lasso Regression to get feature importance
        print(colored('Calculating Importance of features ...', 'cyan'))
        model = LassoRegression(penalty=bp)
        model.fit(X=X_train,Y=Y_train)
        weights = model.weight
        features_importance = []
        feature_names = data.columns
        for idx,coef in enumerate(weights):
            importance = abs(coef)*np.mean(X_train[:,idx])      # Importance of that feature
            features_importance.append((feature_names[idx],importance))

        features_importance.sort(key=lambda x: x[1],reverse=True)

        with open(os.path.join(results_path, "features_importance.txt"),'w') as f:
            f.write('Feature importance from highest to lowest : '+'\n')
            f.write('\n')
            idx = 1
            for feature,importance in features_importance:
                f.write(str(idx)+'. '+feature+' : '+ str(importance) +'\n')
                idx += 1

        print(colored('Importance of features calculated', 'cyan'))