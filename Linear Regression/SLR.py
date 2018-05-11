
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

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Train,Y_Train)

xpred = regressor.predict(X_Test)

py.scatter(X_Train,Y_Train,color = 'red')
py.plot(X_Train,regressor.predict(X_Train),color = 'blue')
py.title('Salary Vs Experience')
py.xlabel('years_of_exp')
py.ylabel('salary')
py.show()

py.scatter(X_Test,Y_Test,color = 'red')
py.plot(X_Train,regressor.predict(X_Train),color = 'blue')
py.title('Salary Vs Experience')
py.xlabel('years_of_exp')
py.ylabel('salary')
py.show()
