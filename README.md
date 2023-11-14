# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.import pandas module and import the required data set.

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
Developed by: monicka s
RegisterNumber:  212221220033
*/
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
Data head:
![image](https://github.com/Monicka19/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/143497806/9b68981a-1907-4886-8f2c-5abefbdd89a6)
Dataset info:
![image](https://github.com/Monicka19/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/143497806/541ffe87-8c9c-4c0f-a466-35deb5ee53e4)
Nulldata set:
![image](https://github.com/Monicka19/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/143497806/052c1875-9972-41e8-a7b9-947fb4422c17)
Values count in left column:
![image](https://github.com/Monicka19/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/143497806/cdc284bc-8504-4ab8-abf9-1ca8eb2a73d8)
Dataset transformed head:
![image](https://github.com/Monicka19/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/143497806/4e6a8920-6ee6-4d72-a678-e7ed12f62f59)
x.head:
![image](https://github.com/Monicka19/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/143497806/70a9daa6-d903-4662-b451-538a904bcd45)
Accuracy:
![image](https://github.com/Monicka19/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/143497806/569bbefb-5de2-4586-ba09-33285b3d5cd1)
Data prediction:
![image](https://github.com/Monicka19/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/143497806/3c83e1d2-f9b5-48cf-af40-e1539f8a5b32)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
