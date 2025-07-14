# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 22:43:34 2025

@author: senaa
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
data = pd.read_csv("C:/Users/senaa/PythonDataScience/SAML-D.csv")

from sklearn.model_selection import train_test_split
data , _ = train_test_split(data, 
                                 train_size=500_000, 
                                 stratify=data['Is_laundering'], 
                                 random_state=42)

data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = pd.to_datetime(data['Date']).dt.year
data['Month'] = pd.to_datetime(data['Date']).dt.month
data['Day'] = pd.to_datetime(data['Date']).dt.day
data['Week'] = data['Date'].dt.isocalendar().week

data.drop(columns=['Time', 'Date'], inplace=True)
numeric_cols = data.select_dtypes(exclude="object").columns
numeric_cols= data[numeric_cols]

categorical_cols = data.select_dtypes(include="object").columns
categorical_data= data[categorical_cols]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
categorical_data['Payment_type_Encoded'] = le.fit_transform(categorical_data['Payment_type'])
ohe = preprocessing.OneHotEncoder()
paymentohe = ohe.fit_transform(categorical_data[['Payment_type_Encoded']]).toarray()
payment = pd.DataFrame(data=paymentohe,index=range(len(paymentohe)),columns=['ACH', 'Cash Deposit', 'Cash Withdrawal', 'Cheque', 'Credit card', 'Cross-border','Debit Card'])
payment = payment.iloc[:,:-1]

categorical_data.drop(columns=['Payment_type_Encoded'], inplace=True)
categorical_data_encoded = categorical_data.copy()

for column in categorical_data_encoded.columns:
    categorical_data_encoded[column] = le.fit_transform(categorical_data_encoded[column])
categorical_final = pd.concat([
    categorical_data_encoded.reset_index(drop=True),
    payment.reset_index(drop=True)
], axis=1)

numeric_cols = pd.concat([numeric_cols.iloc[:,:3],numeric_cols.iloc[:,4:]],axis=1)
finaldata = pd.concat([numeric_cols.reset_index(drop=True),categorical_final.reset_index(drop=True)],axis=1)

x =finaldata.values.astype(float)
y= data.iloc[:,8:9].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print('KNN')
print(cm)

from sklearn.svm import SVC
svc = SVC(kernel='poly')
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)









