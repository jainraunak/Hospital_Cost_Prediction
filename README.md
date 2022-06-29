# Hospital Cost Prediction

In current world, Machine Learning has become an important tool 
and is used in many applications. One of its application is
in Health Care. This project is used to predict **Total Costs** 
of a patient based on patient disease, hospital types, etc. 

### Applications of this Project
- A patient can see whether he/she has been charged properly by the hospital.
- Bases on estimated cost, a patient can decide the hospital to be admitted in. 
- Can see which hospitals are overcharging.

### Data 

The training and test data are subsets of [this dataset](https://healthdata.gov/State/Hospital-Inpatient-Discharges-SPARCS-De-Identified/nff8-2va3). Done ordinal encoding in the original dataset to create train and test data and remove some unwanted features. The encoding of the text can be found at <font color = "blue"> data/Encoding.txt </font>.

### Machine Learning models

Predicting total costs is a regression problem. In this project, 
we are going to use [**Linear Regression**](https://machinelearningmastery.com/linear-regression-for-machine-learning/), 
[**Ridge Regression**](https://online.stat.psu.edu/stat857/node/155/) and 
[**Lasso Regression**](https://www.mygreatlearning.com/blog/understanding-of-lasso-regression/#:~:text=Lasso%20regression%20is%20a%20regularization,i.e.%20models%20with%20fewer%20parameters) to predict total costs.

### Packages to be installed

The packages that are to be installed is written in <font color = "blue"> requirements.txt </font> file.

### Parts 

We divide our project into 3 parts : 

**(a) Part a** : In this part, we will use Linear Regression to predict
total costs. The method used for Linear Regression is **Moore-Penrose pseudoinverse**. I have implemented Linear Regression from scratch and
one can find the implementation in <font color = 'blue'> **src/models/LinearRegression.py** </font> file.

**Implementation file :** <font color = 'blue'> **src/Part_a.py** </font>

**(b) Part b** : In this part, we will use Ridge Regression to predict total
costs. The method used for Ridge Regression is **Moore-Penrose pseudoinverse**. I have implemented Ridge Regression from scratch and one can find the implementation in <font color = 'blue'> **src/models/RidgeRegression.py** </font> file.
Parameters.yml file contain the regularisation penalties that are to be used in ridge regression and we will find the best regularisation penalty among them.

**Implementation file :** <font color = 'blue'> **src/Part_b.py** </font>

**(c) Part c** : In this part, we will use feature engineering to improve our **R<sup>2</sup> Score**. One can look at <font color = 'blue'> **FeatureEngineering.pdf** </font> for more details
about feature engineering. The implementation of feature engineering can be found at <font color = 'blue'> **src/feature_engineering.py** </font> file. 

After feature engineering, we use **Ridge Regression** to predict total costs and find the best regularisation penalty using random search. For finding the importance of these features, I have used **Lasso Regression**.

**Implementation file :** <font color = 'blue'> **src/Part_c.py** </font>

### Parameters 

The <font color = "blue"> Parameters.yml </font> file contains the parameters of this project. One can edit this file based on his/her convenience. 

### Cross Validation 

For finding the best regularisation penalty in ridge regression, I have used k-fold cross validation with **R<sup>2</sup> Score** as a scoring metric. The k for k-fold cross validation is specified in <font color = "blue"> Parameters.yml </font> file.

### Improvement due to feature engineering

**10-fold R<sup>2</sup> Score** without feature engineering was **0.56** and after feature engineering is **0.768**. There is an improvement of **37%** in **10-fold R<sup>2</sup> Score**. 

### How to run this project 

**<u>Step 1</u>** : Open terminal and clone the repository : <br>
**<u>Step 2</u>** : In the terminal, go to the location where the repo is saved. <br>
**<u>Step 3</u>** : Run <font color = "green"> python3 run.py </font> from the terminal. 



  
