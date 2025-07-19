# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 16:38:30 2025

@author: senaa
"""
import pandas as pd 
data = pd.read_csv("C:/Users/senaa/PythonDataScience/Wine.csv")

#Veri Ön İşleme
X = data.iloc[:,0:13].values
Y = data.iloc[:,13].values

#Verilerin Eğitim Ve Test İçin Bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Verilerin Ölçeklenmesi
from  sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2, y_train)

y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)
cm3 = confusion_matrix(y_pred, y_pred2)
print(cm3)