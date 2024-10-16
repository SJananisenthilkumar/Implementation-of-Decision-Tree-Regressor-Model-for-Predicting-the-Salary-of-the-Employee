# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: JANANI S
RegisterNumber:  212223230086
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("/content/Salary (2).csv")
data.head()
```
![image](https://github.com/user-attachments/assets/251bda71-e045-4edb-8baf-b93307cc3fc3)
```
data.info()
```
![image](https://github.com/user-attachments/assets/57c739a1-de9c-4c8c-85ab-9bea7467735a)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/f5c38613-a304-4ead-8968-1367413c9d93)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
```
![image](https://github.com/user-attachments/assets/680781b2-d214-4101-873d-e80fcb52f13d)
```
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(xtrain,ytrain)
y_pred=dt.predict(xtest)
print(y_pred)
```
![image](https://github.com/user-attachments/assets/f97ce0ca-fbdb-439f-943c-26ff6edc2d56)
```
from sklearn import metrics
mse=metrics.mean_squared_error(ytest,y_pred)
print(mse)
```
![image](https://github.com/user-attachments/assets/df345d50-e643-404c-951e-b733e501069d)
```
r2=metrics.r2_score(ytest,y_pred)
print(r2)
```
![image](https://github.com/user-attachments/assets/9d799dfe-4183-4649-add2-d7cc7fc10dd4)
```
dt.predict([[5,6]])
dt.predict([[5,6]])
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()
```

## Output:
![image](https://github.com/user-attachments/assets/7c332472-1de9-41cd-8cb7-f1f3634cd95f)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
