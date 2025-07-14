# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 16:41:35 2025

@author: senaa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
data = pd.read_csv("C:/Users/senaa/PythonDataScience/Ads_CTR_Optimisation.csv")

import random

N = len(data)
D = len(data.columns)
total = 0
chosen = []

for n in range(0,N):
    d = random.randrange(D)
    chosen.append(d)
    prize = data.values[n,d]
    total += prize

plt.hist(chosen)
plt.show()


#Upper Confidence Bound ~ UCB
