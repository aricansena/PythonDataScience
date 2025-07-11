# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 16:31:59 2025

@author: senaa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
data = pd.read_csv("C:/Users/senaa/PythonDataScience/data.csv")

x= data.iloc[:,1:4].values
y= data.iloc[:,-1:].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)

#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train,y_train.ravel())
y_pred = log_reg.predict(X_test)

from sklearn.metrics import confusion_matrix

cm= confusion_matrix(y_test, y_pred)
print("LogisticRegression")
print(cm)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("KNN")
print(cm)

from sklearn.svm import SVC

svc = SVC(kernel="rbf")
svc.fit(X_train, y_train)
p_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("svc")
print(cm)


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("GaussianNB")
print(cm)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)
y_pred= dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("DecisionTreeClassifier")
print(cm)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(X_train, y_train)
y_pred= dtc.predict(X_test)
y_proba = rfc.predict_proba(X_test)

cm = confusion_matrix(y_test,y_pred)
print("RandomForestClassifier")
print(cm)

print(y_proba[:,0])
































