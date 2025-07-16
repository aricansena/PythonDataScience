# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 14:02:20 2025

@author: senaa
"""

import pandas as pd
import numpy as np

data = pd.read_excel("C:/Users/senaa/PythonDataScience/Restaurant_Reviews.xlsx")

data.rename(columns={
    "Column1": "Review",
    "Column2": "Liked"
}, inplace=True)

data = data.iloc[1:, :].reset_index(drop=True)
data["Liked"] = pd.to_numeric(data["Liked"], errors='coerce')
data = data[data["Liked"].isin([0, 1])].reset_index(drop=True)

import re
import nltk

from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()

stopwords = nltk.download('stopwords')
from nltk.corpus import stopwords

#Preprocessing ~~ Önişleme
comments = []
for i in range(704):
    comment = re.sub('[^a-zA-Z]',' ',data['Review'][i])
    comment = comment.lower()
    comment = comment.split()
    comment =[ps.stem(word) for word in comment if word not in set(stopwords.words('english'))]
    comment = ' '.join(comment)
    comments.append(comment)
    
#Feautre Extraction ~~ Öznitelik Çıkarımı 
#Bag of Words BOW
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000)
X= cv.fit_transform(comments).toarray()
y= data.iloc[:,1].values

#Machine Learning
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,stratify=y, random_state=42)

#GaussianNB
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print('GaussianNB')
print(cm)

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4, metric='minkowski')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('KNN')
print(cm)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=25, criterion = 'entropy')
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)










    