# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2.Upload the dataset and check for any null values using .isnull() function.
3.Import LabelEncoder and encode the dataset.
4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5.Predict the values of arrays.
6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7.Predict the values of array.
8.Apply to new unknown values.


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Daniel C
RegisterNumber: 212223240023


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()
*/
```

## Output:
# HEAD(), INFO() & NULL():
![image](https://github.com/Daniel-christal/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145742847/7a8156c1-2e8e-43c9-8533-f9d3d7ddc482)
# CONVERTING STRING LITERALS TO NUMERICAL VALUES USING LABEL ENCODER:
![image](https://github.com/Daniel-christal/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145742847/f724ee9d-cf90-41f1-a8f6-8af7102eca68)
# MEAN SQUARED ERROR:
![image](https://github.com/Daniel-christal/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145742847/6d32b3b9-7d6c-4e70-b2e2-c5d8a8ff0fb9)
# R2 (Variance):
![image](https://github.com/Daniel-christal/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145742847/83f8611f-9efe-46c8-a056-1d53c818acda)
# DATA PREDICTION & DECISION TREE REGRESSOR FOR PREDICTING THE SALARY OF THE EMPLOYEE:
![image](https://github.com/Daniel-christal/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145742847/f20e3f01-899c-426f-9c0d-e39d2785bb60)


## Result:
Thus, the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
