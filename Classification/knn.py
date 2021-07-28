import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv("values.csv")
print(datas)

x = datas.iloc[:,1:4].values #bağımsız değişkenler
y = datas.iloc[:,4:].values #bağımlı değişken
#print(x)
#print(y)

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#k nearest neighborhood
from sklearn.neighbors import KNeighborsClassifier
#komşu sayısı ve yöntemi parametrede seçiyoruz 
#default olarak n_neighbors=5,metric='minkowski'
knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski')
knn.fit(X_train,y_train.ravel())
y_pred = knn.predict(X_test)

from sklearn.metrics import confusion_matrix
conf_m = confusion_matrix(y_test,y_pred)
print(conf_m) 