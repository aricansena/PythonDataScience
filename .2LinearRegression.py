# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 12:42:11 2025

@author: senaa
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("C:/Users/senaa/PythonDataScience/sales.csv")

print(data)

months = data[["months"]]
print(months)

sales = data[["sales"]]
print(sales)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(months,sales,test_size=0.33,random_state=0)

#Doğrusal Regresyon - Model İnşası

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

predict = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))