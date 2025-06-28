# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 23:21:22 2025

@author: senaa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
 
data = pd.read_csv("C:/Users/senaa/PythonDataScience/sales_new.csv")

data_numeric = data.iloc[:,2:]

x= data.iloc[:,2:3]
y= data.iloc[:,-1:]

X = x.values
Y = y.values

print(data_numeric.corr())

"""
               UnvanSeviyesi     Kidem      Puan      maas
UnvanSeviyesi       1.000000 -0.125200  0.034948  0.727036
Kidem              -0.125200  1.000000  0.322796  0.117964
Puan                0.034948  0.322796  1.000000  0.201474
maas                0.727036  0.117964  0.201474  1.000000

"""

#Linear Regression
from sklearn.linear_model import LinearRegression
lr_reg = LinearRegression()
lr_reg.fit(X, Y)

import statsmodels.api as sm
print("Linear OLS")
model =sm.OLS(lr_reg.predict(X),X)
print(model.fit().summary())

print("Linear Regression R Square")
print(r2_score(Y, lr_reg.predict(X)))

"""
plt.figure(figsize=(3,2))
plt.scatter(x, y, color="black") #Gerçek Değerler
plt.plot(x, lr_reg.predict(x), color="blue") #Tahmin Doğrusu
plt.title("Linear Regression")
plt.show()
"""
#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lr_poly = LinearRegression()
lr_poly.fit(x_poly,y)

print("Poly OLS")
model2 =sm.OLS( lr_poly.predict(poly_reg.fit_transform(x)),x)
print(model2.fit().summary())

print("Polynomial R Square")
print(r2_score(y, lr_poly.predict(poly_reg.fit_transform(x))))

"""
plt.figure(figsize=(3,2))
plt.scatter(x, y, color="red")
plt.plot(x, lr_poly.predict(poly_reg.fit_transform(x)), color="green")
plt.title("Polynomial Regression")
plt.show()
"""
##########SVR###########

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X= sc.fit_transform(x)
sc2= StandardScaler()
Y = sc2.fit_transform(y)

from sklearn.svm import SVR
svr_reg = SVR (kernel="rbf") # Linear,Polynomial,Guassian(RBF),Exponential(Üssel)
svr_reg.fit(X, Y)

print("SVR OLS")
model3 =sm.OLS(svr_reg.predict(X),X)
print(model3.fit().summary())

print("SVR R Square")
print(r2_score(Y, svr_reg.predict(X)))
"""
plt.figure(figsize=(3,2))
plt.scatter(X, Y,color="red")
plt.plot(X, svr_reg.predict(X))
plt.title("SVR")
plt.show()
"""

from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(x, y)

print("Decision Tree OLS")
model4 =sm.OLS(dt_reg.predict(x),x)
print(model4.fit().summary())

print("Decision Tree R Square")
print(r2_score(y, dt_reg.predict(x)))
"""
plt.scatter(x, y,color="red")
plt.plot(x,dt_reg.predict(x),color="blue")
plt.title("Decision Tree")
plt.show()
"""

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(x,y)

print("Random Forest OLS")
model5 =sm.OLS(rf_reg.predict(x),x)
print(model5.fit().summary())

print("Random Forest R Square")
print(r2_score(y, rf_reg.predict(x)))

"""
plt.scatter(x, y, color="red")
plt.plot(x,rf_reg.predict(x),color="blue")
plt.title("Random Forest")
plt.show()
"""

print("Linear Regression R Square")
print(r2_score(y, lr_reg.predict(x)))

print("Polynomial R Square")
print(r2_score(y, lr_poly.predict(poly_reg.fit_transform(x))))

print("SVR R Square")
print(r2_score(Y, svr_reg.predict(X)))

print("Decision Tree R Square")
print(r2_score(y, dt_reg.predict(x)))

print("Random Forest R Square")
print(r2_score(y, rf_reg.predict(x)))

"""
OLS
linear
R-squared (uncentered):                   0.942

Polynomial
R-squared (uncentered):                   0.759

SVR
R-squared (uncentered):              0.770

Decision Tree
R-squared (uncentered):                   0.751

Random Forest 
R-squared (uncentered):                   0.719
"""

""" 
(OLS modeli): polinom tahminlerinin x ile açıklanabilirliğini ölçüyor (pek anlamlı bir performans ölçütü değil).
(r2_score): polinom modelin gerçek y verisini ne kadar açıkladığını gösteriyor, bu senin asıl takip etmen gereken R².
"""