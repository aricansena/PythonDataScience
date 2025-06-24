# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 22:09:53 2025

@author: senaa
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
data = pd.read_csv("C:/Users/senaa/PythonDataScience/titlesales.csv")

#Linear Regression
x= data.iloc[:,1:2]
y= data.iloc[:,-1:]

X= x.values
Y = y.values

from sklearn.linear_model import LinearRegression
lr_linear = LinearRegression()
lr_linear.fit(X, Y)

plt.figure(figsize=(3,2))
plt.scatter(X, Y, color="red") #Gerçek Değerler
plt.plot(x, lr_linear.predict(X), color="blue") #Tahmin Doğrusu
plt.xlabel("Bağımsız Değişken")
plt.ylabel("Bağımlı Değişken")
plt.title("Linear Regression")

#Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)

lr_poly = LinearRegression()
lr_poly.fit(x_poly,y)
plt.figure(figsize=(3,2))
plt.scatter(X, Y, color="red")
plt.plot(X, lr_poly.predict(poly_reg.fit_transform(X)), color="blue")
plt.title("Polynomial Regression")

print(lr_linear.predict([[6.6]]))
print(lr_linear.predict([[11]]))
print(lr_poly.predict(poly_reg.fit_transform([[6.6]])))
print(lr_poly.predict(poly_reg.fit_transform([[11]])))
