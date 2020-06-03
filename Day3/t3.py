import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
dataframe = pd.read_csv('http://iali.in/datasets/mushrooms.csv')
labelencoder = LabelEncoder()
for col in dataframe.columns:
    dataframe[col] = labelencoder.fit_transform(dataframe[col])

X = dataframe.drop(columns=['class'])
y = dataframe.drop(columns=["cap-shape","cap-surface","cap-color","bruises","odor","gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring","veil-type","veil-color","ring-number","ring-type","spore-print-color","population","habitat"])

# y = dataframe.iloc[:, 0:1].value
# x = dataframe.iloc[:, 1:].values
# print(x)
X_train, X_test, Y_train, Y_test = train_test_split(X, y.values.ravel(), test_size=0.2, random_state=10)

log = LogisticRegression()
log.fit(X_train, Y_train)
prediction=log.predict(X_test)
print(prediction)
print (accuracy_score(Y_test, prediction))
matrix = confusion_matrix(Y_test, prediction)
print(matrix)
report = classification_report(Y_test, prediction)
print(report)
