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
#fit_transform yerine transform kullanırsak yeniden öğrenmiyor öğrenileni kullanıyor

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train,y_train)

y_pred = log_reg.predict(X_test)
print("Tahmin:",y_pred,"\n Veriler: \n",y_test)

#confusion matrix
#sınıflandırmanın doğruluğunu ölçüyoruz
from sklearn.metrics import confusion_matrix
conf_m = confusion_matrix(y_test,y_pred)
print(conf_m)