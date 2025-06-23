# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 16:09:16 2025

@author: senaa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
data = pd.read_csv("C:/Users/senaa/PythonDataScience/tennis.csv")

from sklearn import preprocessing
data2 = data.apply(preprocessing.LabelEncoder().fit_transform)

c = data2.iloc[:,:1]
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)

outlook = pd.DataFrame(data=c,index=range(14),columns=["overcast","rainy","sunny"])
lastdata = pd.concat([outlook,data.iloc[:,1:3]],axis=1)

lastdata = pd.concat([data2.iloc[:,3:],lastdata],axis=1)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(lastdata.iloc[:,:-1],lastdata.iloc[:,-1:],test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

import statsmodels.api as sm

X= np.append(arr= np.ones((14,1)).astype(int), values=lastdata.iloc[:,:-1],axis=1)

X_l = lastdata.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(lastdata.iloc[:,-1:],X_l).fit()
print(model.summary())

lastdata =lastdata.iloc[:,1:]

X_l = lastdata.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(lastdata.iloc[:,-1:],X_l).fit()
print(model.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

x_train,x_test,y_train,y_test = train_test_split(lastdata.iloc[:,:-1],lastdata.iloc[:,-1:],test_size=0.33,random_state=0)

lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

"""
#########

outlook = data.iloc[:,0:1].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

outlook[:,0] = le.fit_transform(data.iloc[:,0])

ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
print(outlook)

windy = data.iloc[:,3:4].values
windy = ohe.fit_transform(windy).toarray()
print(windy)

play = data.iloc[:,-1:].values
play = ohe.fit_transform(play).toarray()
print(play)

outlook = pd.DataFrame(data=outlook,index=range(14),columns=["overcast","rainy","sunny"])

windy = pd.DataFrame(data=windy[:,1:2],index=range(14),columns=["windy"])

play = pd.DataFrame(data=play[:,1:2],index=range(14),columns=["play"])

therest= pd.DataFrame(data=data.iloc[:,1:3],index=range(14),columns=["temperature","humidity"])

s1= pd.concat([outlook,therest],axis=1)
print(s1)

s2= pd.concat([s1,windy],axis=1)
print(s2)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s2,play,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

import statsmodels.api as sm

X= np.append(arr= np.ones((14,1)).astype(int), values=s2,axis=1)

X_l = s2.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(play,X_l).fit()
print(model.summary())

X_l = s2.iloc[:,[0,1,2,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(play,X_l).fit()
print(model.summary())

"""











