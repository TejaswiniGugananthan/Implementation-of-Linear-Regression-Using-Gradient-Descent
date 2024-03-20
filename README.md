# Implementation-of-Linear-Regression-Using-Gradient-Descent
## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

### Steps involved: 

1.Data Preparation: The first step is to prepare the data for the model. This involves cleaning the data, handling missing values and outliers, and transforming the data into a suitable format for the model.

2.Split the data: Split the data into training and testing sets. The training set is used to fit the model, while the testing set is used to evaluate the model's performance.

3.Define the model: The next step is to define the logistic regression model. This involves selecting the appropriate features, specifying the regularization parameter, and defining the loss function.

4.Train the model: Train the model using the training data. This involves minimizing the loss function by adjusting the model's parameters.

5.Evaluate the model: Evaluate the model's performance using the testing data. This involves calculating the model's accuracy, precision, recall, and F1 score.

6.Tune the model: If the model's performance is not satisfactory, you can tune the model by adjusting the regularization parameter, selecting different features, or using a different algorithm.

7.Predict new data: Once the model is trained and tuned, you can use it to predict new data. This involves applying the model to the new data and obtaining the predicted outcomes.

8.Interpret the results: Finally, you can interpret the model's results to gain insight into the relationship between the input variables and the output variable. This can help you understand the factors that influence the outcome and make informed decisions based on the results.

## Program:

Program to implement the linear regression using gradient descent.

Developed by: G.TEJASWINI

RegisterNumber:  212222230157


```python
import pandas as pd
df=pd.read_csv("Placement_Data.csv")
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x=df1.iloc[:,:-1]
x

y=df1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = Ir. predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

Ir.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```




## Output:
1. Placememt data:

<img width="466" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121222763/87422c8b-c8e4-477f-8de5-4f0bc463ce78">

2. Salary data:

<img width="415" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121222763/f50388da-e919-488a-a235-cf77ed0c431f">

3. Checking the null function:

<img width="86" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121222763/bae81826-39b0-4bd6-8c14-819db6b721c7">

4. Data duplicate:
   
<img width="20" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121222763/288f391d-1daa-4645-bb29-cbf33343910d">

5.Print data:

<img width="388" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121222763/abd3650f-50d3-46e6-8664-6aca5d244528">

6. Data status:
   
<img width="164" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121222763/ceda8e70-fe59-4217-9b60-1015d8d539ce">

7. Y-Prediction array:
   
<img width="295" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121222763/16628e30-08c6-40e0-9d03-25093f9b7375">

8. Accuray value:
 
<img width="187" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121222763/0613b0fc-8649-4210-b844-dddd9c71c91b">

9. Confusion matrix:
    
<img width="142" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121222763/07530f49-6cdb-4702-977b-1c75ee47882c">

10. Classification Report:

<img width="226" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121222763/1a390bd9-b1bf-4cf5-9e6d-410831ada8e9">

11. Prediction of LR:

<img width="503" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121222763/a71e42c6-55b8-4f6c-b1db-82893d8e9f68">


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
