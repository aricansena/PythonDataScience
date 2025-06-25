# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 17:12:44 2025

@author: senaa
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
data = pd.read_csv("C:/Users/senaa/PythonDataScience/titlesales.csv")

x= data.iloc[:,1:2].values
y= data.iloc[:,-1:].values

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X= sc.fit_transform(x)
sc2= StandardScaler()
Y = sc2.fit_transform(y)

from sklearn.svm import SVR
svr_reg = SVR (kernel="rbf") # Linear,Polynomial,Guassian(RBF),Exponential(Üssel)
svr_reg.fit(X, Y)
#plt.figure(figsize=(3,2))
plt.scatter(X, Y,color="red")
plt.plot(X, svr_reg.predict(X))
plt.show()


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=0)

dt.fit(x, y)
plt.scatter(x, y,color="green")
plt.plot(x,dt.predict(x),color="blue")
plt.show()
