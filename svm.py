import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

df=pd.read_csv('data2.csv')
df=df[:1000]
X = df.drop(['Unnamed: 0','priority','scheduling delay', 'failed'], axis=1)
Y = df['failed']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.25,stratify=Y)
classifier = svm.SVC(kernel='poly')
classifier.fit(X_train,Y_train)

##train_predict=classifier.predict(X_train)
##acc=accuracy_score(train_predict,Y_train)

test_predict=classifier.predict(X_test)
acc1=accuracy_score(test_predict,Y_test)

print('{}'.format(acc1))

