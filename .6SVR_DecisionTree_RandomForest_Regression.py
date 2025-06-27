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
plt.title("SVR")
plt.show()


from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(x, y)
Z= x + 0.5
K = x - 0.4
plt.scatter(x, y,color="red")
plt.plot(x,dt_reg.predict(x),color="blue")
plt.plot(x,dt_reg.predict(Z),color="green")
plt.plot(x,dt_reg.predict(K),color="yellow")
plt.title("Decision Tree")
plt.show()

#DecisionTree x,Z,K bu üç değer için aynı tahmini döndürür çünkü gelmiş olduğu aralığa göre aynı değere indirger

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(x,y.ravel())
print(rf_reg.predict([[6.5]]))

plt.scatter(x, y, color="red")
plt.plot(x,rf_reg.predict(x),color="blue")
plt.plot(x,rf_reg.predict(Z),color="green")
plt.plot(x,rf_reg.predict(K),color="yellow")
plt.title("Random Forest")
plt.show()
from sklearn.metrics import r2_score

print("SVR R Square")
print(r2_score(Y, svr_reg.predict(X)))

print("Decision Tree R Square")
print(r2_score(y, dt_reg.predict(x)))

print("Random Forest R Square")
print(r2_score(y, rf_reg.predict(x)))
