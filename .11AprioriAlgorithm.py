# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 17:59:41 2025

@author: senaa
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
data = pd.read_csv("C:/Users/senaa/PythonDataScience/sepet.csv",header=None)
print(data.shape)

t = []
for i in range(0,7501):
    t.append([str(data.values[i,j])for j in range(0,20)])

from apriori import runApriori

rules = runApriori(t, minSupport=0.01, minConfidence=0.2)

print(list(rules))
