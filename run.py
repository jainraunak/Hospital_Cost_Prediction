from src.Part_a import solve_a
from src.Part_b import solve_b
from src.Part_c import solve_c
from termcolor import colored
from verify.verify_part_a import verify_part_a
from verify.verify_part_b import verify_part_b
import numpy as np
import os
import pandas as pd
import time
import yaml
import warnings
warnings.filterwarnings("ignore")

os.system('color')
# Read Parameters.yml file
with open("Parameters.yml",'r') as stream:
    dic = yaml.safe_load(stream)

# Get training and test data
train_data = pd.read_csv(dic['train_data'])
test_data = pd.read_csv(dic['test_data'])

# Total Costs is the last column of training and test data
Y_train = np.asarray(train_data['Total Costs'])
X_train = np.asarray(train_data.iloc[:,1:-1])
X_test = np.asarray(test_data.iloc[:,1:])

# Adding column of ones to handle the bias
X_train = np.c_[np.ones(X_train.shape[0]),X_train]
X_test = np.c_[np.ones(X_test.shape[0]),X_test]

regc = dic['regularisation_penalty']
results_path = dic['results_path']
k = dic['k']
get_features_importance = dic['get_features_importance']
verify_res = dic['verify_results']
reg_lower_limit = dic['regularisation_penalty_lower_limit']
reg_upper_limit = dic['regularisation_penalty_upper_limit']
random_searches = dic['random_searches']

if "a" in dic['parts']:
    # Do Part a
    start = time.time()
    print(colored('Part a started ... ','magenta'))
    solve_a(X_train=X_train,Y_train=Y_train,X_test=X_test,results_path=results_path)
    if verify_res:
        verify_part_a()
    end = time.time()
    print(colored('Part a finished (' + str(round(end-start,2))+' s)', 'magenta'))

if "b" in dic['parts']:
    # Do Part b
    start = time.time()
    print(colored('Part b started ... ', 'magenta'))
    solve_b(X_train=X_train, Y_train=Y_train, X_test=X_test, results_path=results_path,regc=regc,k=k)
    if verify_res:
        verify_part_b()
    end = time.time()
    print(colored('Part b finished (' + str(round(end-start,2))+' s)', 'magenta'))

if "c" in dic['parts']:
    # Do Part c
    start = time.time()
    print(colored('Part c started ... ', 'magenta'))
    solve_c(train_data=train_data,test_data=test_data,results_path=results_path,k=k,
            get_features_importance=get_features_importance,reg_lower_limit=reg_lower_limit,
            reg_upper_limit=reg_lower_limit,random_searches=random_searches)
    end = time.time()
    print(colored('Part c finished (' + str(round(end-start,2))+' s)', 'magenta'))