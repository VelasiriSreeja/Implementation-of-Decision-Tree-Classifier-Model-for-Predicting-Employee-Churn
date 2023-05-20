# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2.Find the null values and count them.
3.Count number of left values.
4.From sklearn import LabelEncoder to convert string values to numerical values.
5.From sklearn.model_selection import train_test_split.
6.Assign the train dataset and test dataset.
7.From sklearn.tree import DecisionTreeClassifier.
8.Use criteria as entropy.
9.From sklearn import metrics.
10.Find the accuracy of our model and predict the require values. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: v.sreeja
RegisterNumber:  212222230169

import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

Data.head()

![Screenshot (228)](https://github.com/VelasiriSreeja/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118344328/401af0d5-c662-49ba-8223-f504af4f1e04)

Data.info()

![Screenshot (229)](https://github.com/VelasiriSreeja/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118344328/029bfd8c-f6f6-4e8e-b813-542bb981c8df)

isnull() and sum()

![Screenshot (230)](https://github.com/VelasiriSreeja/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118344328/27a1dd28-f427-442b-a432-3e4290205046)

Data value Counts()

![Screenshot (231)](https://github.com/VelasiriSreeja/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118344328/27fff9e4-28ce-4b24-862e-e9419b754cbc)

Data.head() for salary

![Screenshot (232)](https://github.com/VelasiriSreeja/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118344328/d5154e3f-27f1-4bc7-b003-d815bb6cb98f)

x.head()

![Screenshot (233)](https://github.com/VelasiriSreeja/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118344328/19cd9770-b8a0-4a83-b936-b789de512118)

Accuracy Value:

![Screenshot (234)](https://github.com/VelasiriSreeja/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118344328/db71f4ec-740a-4e8c-9a35-3d38e1f9f62c)

Data Predicton:

![Screenshot (235)](https://github.com/VelasiriSreeja/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118344328/c80a4dcc-91e1-4170-95ed-7708f80d02fe)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
