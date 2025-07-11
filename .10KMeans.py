# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 12:22:54 2025

@author: senaa
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('C:/Users/senaa/PythonDataScience/customers.csv')
X = data.iloc[:,3:].values

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,init='k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)
results=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=123)
    kmeans.fit(X)
    results.append(kmeans.inertia_)
    
plt.plot(range(1,11),results)
plt.show()

kmeans = KMeans(n_clusters=3,init='k-means++')
y_pred=kmeans.fit_predict(X)
plt.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100,c='red')
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100,c='blue')
plt.scatter(X[y_pred==2,0],X[y_pred==2,1],s=100,c='green')
plt.show()


#HC
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3,metric='euclidean',linkage='ward')

y_pred= ac.fit_predict(X)
print(y_pred)

plt.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100,c='red')
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100,c='blue')
plt.scatter(X[y_pred==2,0],X[y_pred==2,1],s=100,c='green')
plt.show()



import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.show()





