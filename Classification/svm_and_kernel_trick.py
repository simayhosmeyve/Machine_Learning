import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv("values.csv")

x = datas.iloc[:,1:4].values 
y = datas.iloc[:,4:].values 

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.svm import SVC
#svc = SVC(kernel='linear')
#svc = SVC(kernel='rbf')
#farklı kernel seçimlerini test ediyoruz

#kernel trick
#doğrusal olarak sınıflandıramayacağımız verilerde boyut arttırıyoruz
#kernel seçimi sınıflandırmayı değiştirir
svc = SVC(kernel='sigmoid') 
svc.fit(X_train,y_train)
y_predict = svc.predict(X_test)
print(y_predict)
print(y_test)