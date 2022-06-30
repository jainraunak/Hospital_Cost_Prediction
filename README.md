# Hospital Cost Prediction

In current world, Machine Learning has become an important tool 
and is used in many applications. One of its application is
in Health Care. This project is used to predict **Total Costs** 
of a patient based on patient disease, hospital types, etc. 

### Applications of this Project
- A patient can see whether he/she has been charged properly by the hospital.
- Based on estimated cost, a patient can decide the hospital to be admitted in. 
- Can see which hospitals are overcharging.

### Data 

The training and test data are subsets of [**this dataset**](https://healthdata.gov/State/Hospital-Inpatient-Discharges-SPARCS-De-Identified/nff8-2va3). Done ordinal encoding in the original dataset to create train and test data and remove some unwanted features. The encoding of the text can be found at [**Encoding.txt**](data/Encoding.txt) </font>.

### Machine Learning models

Predicting total costs is a regression problem. In this project, 
we are going to use [**Linear Regression**](https://machinelearningmastery.com/linear-regression-for-machine-learning/), 
[**Ridge Regression**](https://online.stat.psu.edu/stat857/node/155/) and 
[**Lasso Regression**](https://www.mygreatlearning.com/blog/understanding-of-lasso-regression/#:~:text=Lasso%20regression%20is%20a%20regularization,i.e.%20models%20with%20fewer%20parameters) to predict total costs.

### Packages to be installed

The packages that are to be installed is written in [**requirements.txt**](requirements.txt) file.

### Parts 

We divide our project into 3 parts : 

**(a) Part a** : In this part, we will use Linear Regression to predict
total costs. The method used for Linear Regression is [**Moore-Penrose pseudoinverse**](https://www.geeksforgeeks.org/moore-penrose-pseudoinverse-mathematics/). I have implemented Linear Regression from scratch and
one can find the implementation in [**LinearRegression.py**](src/models/LinearRegression.py) file.

**Implementation file :** [**Part_a.py**](src/Part_a.py) 

**(b) Part b** : In this part, we will use Ridge Regression to predict total
costs. The method used for Ridge Regression is [**Moore-Penrose pseudoinverse**](https://www.geeksforgeeks.org/moore-penrose-pseudoinverse-mathematics/). I have implemented Ridge Regression from scratch and one can find the implementation in [**RidgeRegression.py**](src/models/RidgeRegression.py) file.
Parameters.yml file contain the regularisation penalties that are to be used in ridge regression and we will find the best regularisation penalty among them.

**Implementation file :** [**Part_b.py**](src/Part_b.py) 

**(c) Part c** : In this part, we will use feature engineering to improve our **R<sup>2</sup> Score**. One can look at [**FeatureEngineering.pdf**](FeatureEngineering.pdf) for more details
about feature engineering. The implementation of feature engineering can be found at [**feature_engineering.py**](src/feature_engineering.py) file. 

After feature engineering, we use **Ridge Regression** to predict total costs and find the best regularisation penalty using random search. For finding the importance of these features, I have used **Lasso Regression**.

**Implementation file :** [**Part_c.py**](src/Part_c.py) 

### Parameters 

The [**Parameters.yml**](Parameters.yml) file contains the parameters of this project. One can edit this file based on his/her convenience. 

### Cross Validation 

For finding the best regularisation penalty in ridge regression, I have used k-fold cross validation with **R<sup>2</sup> Score** as a scoring metric. The k for k-fold cross validation is specified in [**Parameters.yml**](Parameters.yml) file.

**Implementation file :** [**CrossValidation.py**](src/CrossValidation.py)

### Improvement due to feature engineering

**10-fold R<sup>2</sup> Score** without feature engineering was **0.56** and after feature engineering is **0.778**. There is an improvement of **38.75%** in **10-fold R<sup>2</sup> Score**. 

### How to run this project 

**<u>Step 1</u>** : Open terminal and clone the repository : git clone https://github.com/jainraunak/Hospital_Cost_Prediction.git <br>
**<u>Step 2</u>** : In the terminal, go to the location where the repo is saved. <br>
**<u>Step 3</u>** : Run **<i>python3 run.py</i>** from the terminal. 



  
