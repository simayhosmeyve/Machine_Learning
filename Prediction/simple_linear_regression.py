import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datas = pd.read_csv("sales.csv")
print(datas)

#verileri ayırıyoruz
months = datas[['Aylar']]
sales = datas[['Satislar']]
#print(months)
#print(sales)

from sklearn.model_selection import train_test_split
#aylar bağımsız değişken, satışlar bağımlı değişken
#ilk önce months yazılmalı
x_train, x_test, y_train, y_test = train_test_split(months,sales,test_size=0.33,random_state=0)
print("Train and Test \n",x_train,"\n",y_train,"\n",x_test,"\n",y_test)

from sklearn.preprocessing import StandardScaler
#standartize ediliyor
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

#veri hazırlık sürecinden sonra artık modelimizi oluşturuyoruz
#Lineer Regresyon
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

#burada çıkan tahmin y_test'e yakın olmalı
prediction = lr.predict(x_test)
print(prediction)

#veriyi görselleştirme
#karışık halde bulunan verileri düzgün bir grafik çıkması için sıralıyoruz
x_train = x_train.sort_index()
y_train =y_train.sort_index()

plt.plot(x_train,y_train)
#plt.show()
#tahmin ve gerçek verilerin grafiği
plt.plot(x_test,lr.predict(x_test))
plt.title("Aylara göre satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")
plt.show()