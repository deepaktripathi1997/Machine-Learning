# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 08:04:24 2018

@author: deepak
"""

import numpy as np
import matplotlib.pyplot as py
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

#training sets and test set
from sklearn.cross_validation import train_test_split
X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y,test_size = 1/3 ,random_state = 0)




"""from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_Train = sc.fit_transform(X_Train)
X_Test = sc.transform(X_Test)"""