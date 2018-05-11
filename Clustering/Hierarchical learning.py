# -*- coding: utf-8 -*-
"""
Created on Wed May  9 07:15:40 2018

@author: deepak
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:,[3,4]].values

#dendrogram
from scipy.cluster import hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean_Distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5,affinity = 'euclidean',linkage = 'ward')
y_hc = hc.fit_predict(X)

#don't use it if there are more than 2 dimensions
plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1],s = 100,c = 'red',label = 'cluster1')
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1],s = 100,c = 'blue',label = 'cluster2')
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1],s = 100,c = 'green',label = 'cluster3')
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1],s = 100,c = 'cyan',label = 'cluster4')
plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1],s = 100,c = 'magenta',label = 'cluster5')
plt.title('cluster')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()


