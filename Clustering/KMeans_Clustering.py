# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:,[3,4]].values

#finding the number of optimum clusters

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('The class levels')
plt.ylabel('wcss')
plt.show

kmeans = KMeans(n_clusters = 5,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)
y_means = kmeans.fit_predict(X)


plt.scatter(X[y_means == 0,0],X[y_means == 0,1],s = 100,c = 'red',label = 'cluster1')
plt.scatter(X[y_means == 1,0],X[y_means == 1,1],s = 100,c = 'blue',label = 'cluster2')
plt.scatter(X[y_means == 2,0],X[y_means == 2,1],s = 100,c = 'green',label = 'cluster3')
plt.scatter(X[y_means == 3,0],X[y_means == 3,1],s = 100,c = 'cyan',label = 'cluster4')
plt.scatter(X[y_means == 4,0],X[y_means == 4,1],s = 100,c = 'magenta',label = 'cluster5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s = 300,c = 'yellow',label = 'Cluster')
plt.title('cluster')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()


