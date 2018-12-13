import pandas as pd
import numpy as np
import pydotplus
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.model_selection import train_test_split

sumberdata = pd.read_csv('pima-indians.data.csv')
print(sumberdata.head(10))
sumberdata.isnull().sum()
sumberdata[['class','ntp','plas','dbp','tsft','serins','mass','dpf','age']].sample(10)
prediktor = sumberdata[['ntp','plas','dbp','tsft','serins','mass','dpf','age']]
target = sumberdata['class']
X_train, X_test, y_train, y_test = train_test_split(prediktor, target, test_size = .2, random_state = 0)

print('X_train = ',X_train.shape)
print('X_test = ',X_test.shape)
print('y_train = ',y_train.shape)
print('y_test = ',y_test.shape)

klasifikasi = DecisionTreeClassifier()
klasifikasi.fit(X_train, y_train)
prediksi = klasifikasi.predict(X_test)
konfusi = sklearn.metrics.confusion_matrix(y_test,prediksi)

print(konfusi)
print(konfusi[0,0])
print(konfusi[1,0])
print(konfusi[1,1])
print(konfusi[0,1])

akurasi = sklearn.metrics.accuracy_score(y_test, prediksi)
print(akurasi)


