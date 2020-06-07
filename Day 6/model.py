import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
dataset = pd.read_csv('http://iali.in/datasets/Social_Network_Ads.csv')
# col = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
# dataset = pd.DataFrame(col.fit_transform(dataset))
label = LabelEncoder()
dataset['Gender'] = label.fit_transform(dataset['Gender'])
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=10)
log = LogisticRegression()

log.fit(X_train, Y_train)
print(X_test)
print(log.predict(X_test))
pickle.dump(log, open('model.pkl','wb'))
model = pickle.load(open('model.pkl', 'rb'))