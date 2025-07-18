# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 17:12:17 2025

@author: senaa
"""
import pandas as pd
import numpy as np
 
data = pd.read_csv("C:/Users/senaa/PythonDataScience/Churn_Modelling.csv")

#Veri Ön İşleme
X = data.iloc[:,3:13].values
Y = data.iloc[:,13].values

# Encoder Kategorik > Numeric
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

legender = preprocessing.LabelEncoder()
X[:,2] = legender.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])], remainder="passthrough")
X = ohe.fit_transform(X)
X = X[:,1:]

#Verilerin Eğitim Ve Test İçin Bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33,stratify=Y, random_state=42)

# Verilerin Ölçeklenmesi
from  sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Yapay Sinir Ağı
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(6, kernel_initializer="uniform",activation="relu",input_dim=11))
classifier.add(Dense(6, kernel_initializer="uniform",activation="relu"))
classifier.add(Dense(1, kernel_initializer="uniform",activation="sigmoid"))
classifier.compile(optimizer='adam',loss= 'binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,y_train,epochs=50)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
