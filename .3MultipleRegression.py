# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 20:45:14 2025

@author: senaa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
data = pd.read_csv("C:/Users/senaa/PythonDataScience/data.csv")

result = data.iloc[:,1:4].values
print(result)

country = data.iloc[:,0:1].values
print(country)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
country[:,0] = le.fit_transform(data.iloc[:,0])
ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
print(country)
 
# gender
gender = data.iloc[:,-1:].values
print(gender)

le = preprocessing.LabelEncoder()
gender[:,-1] = le.fit_transform(data.iloc[:,-1])
ohe = preprocessing.OneHotEncoder()
gender = ohe.fit_transform(gender).toarray()
print(gender)

#numpy dizileri dataFrame Dönüşümü

countryResult = pd.DataFrame(data=country,index=range(22),columns=["fr","tr","us"])
print(countryResult)
result = pd.DataFrame(data=result,index=range(22),columns=["height","weight","age"])
print(result)
genderResult = pd.DataFrame(data=gender[:,:1],index=range(22),columns=["cinsiyet"])
print(genderResult)

# dataframe Birleştirme İşlemi
s =pd.concat([countryResult,result],axis=1)
print(s)
finalResult = pd.concat([s,genderResult],axis=1)
print(finalResult)

#train test split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s,genderResult,test_size=0.33,random_state=0)

#Model İnşası
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

###############################

height = finalResult.iloc[:,3:4].values

s1 = finalResult.iloc[:,:3]
print(s1)
s2= finalResult.iloc[:,4:]
print(s2)


s3= pd.concat([s1,s2],axis=1)
print(s3)

x_train,x_test,y_train,y_test = train_test_split(s3,height,test_size=0.33,random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_predict = regressor.predict(x_test)

################################

#Backward Elimination

import statsmodels.api as sm

X= np.append(arr= np.ones((22,1)).astype(int), values=s3,axis=1)

X_l = s3.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(height,X_l).fit()
print(model.summary())

X_l = s3.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(height,X_l).fit()
print(model.summary())


X_l = s3.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(height,X_l).fit()
print(model.summary())




























































































