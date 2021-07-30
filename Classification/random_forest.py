import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv("values.csv")
x = datas.iloc[:,1:4].values #bağımsız değişkenler
y = datas.iloc[:,4:].values #bağımlı değişken

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
randfc = RandomForestClassifier(n_estimators=5,criterion='entropy')
randfc.fit(X_train,y_train)
y_pred = randfc.predict(X_test)
print(y_pred)
print(y_test)