# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 20:22:10 2025

@author: senaa
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

print(data)

height = data[["height"]]
print(height)

