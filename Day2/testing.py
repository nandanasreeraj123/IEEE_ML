import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
dataset = pd.read_csv('http://iali.in/datasets/Social_Network_Ads.csv')
col = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
dataset = pd.DataFrame(col.fit_transform(dataset))
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=10)

log = LogisticRegression()
log.fit(X_train, Y_train)
prediction=log.predict(X_test)
print(prediction)
print (accuracy_score(Y_test, prediction))
