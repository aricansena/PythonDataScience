# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 21:08:08 2025

@author: senaa
"""

import pandas as pd
import numpy as np
 
data = pd.read_csv("C:/Users/senaa/PythonDataScience/data.csv")

print(data)

height = data[["height"]]
print(height)

heightweight = data[["height","weight"]]
print(heightweight)

class people:
    height = 180
    def run(self,b):
        return b+10
    
Sena = people()
print(Sena.height)

missingData = pd.read_csv("C:/Users/senaa/PythonDataScience/missingdata.csv")
print(missingData)
"""
"""

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")

Age = missingData.iloc[:,1:4].values
print(Age)
imputer = imputer.fit(Age[:,1:4])
Age[:,1:4]= imputer.transform(Age[:,1:4])
print(Age)

country = missingData.iloc[:,0:1].values
print(country)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

country[:,0] = le.fit_transform(missingData.iloc[:,0])

print(country)

ohe = preprocessing.OneHotEncoder()

country = ohe.fit_transform(country).toarray()
print(country)


countryResult = pd.DataFrame(data=country,index=range(22),columns=["fr","tr","us"])
print(countryResult)

ageResult = pd.DataFrame(data=Age,index=range(22),columns=["height","weight","age"])
print(ageResult)

gender = missingData.iloc[:,-1].values
print(gender)

genderResult = pd.DataFrame(data=gender,index=range(22),columns=["gender"])
print(genderResult)


s =pd.concat([countryResult,ageResult],axis=1)
print(s)

finalResult = pd.concat([s,genderResult],axis=1)
print(finalResult)






























