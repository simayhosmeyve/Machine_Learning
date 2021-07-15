#tahminimiz birden fazla parametreye bağlı olduğunda
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv('values.csv')
#print(datas)
age = datas.iloc[:,1:4].values
#print(age)

#encoding
country = datas.iloc[:,0:1].values
print(country)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
country[:,0] = le.fit_transform(datas.iloc[:,0])
print(country)

ohe = preprocessing.OneHotEncoder(categories='auto')
country = ohe.fit_transform(country).toarray()
print(country)

c = datas.iloc[:,-1:].values
print(c)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
c[:,-1] = le.fit_transform(datas.iloc[:,-1])
print(c)


#numpy dizileri dataframe donusumu
df = pd.DataFrame(data=country, index = range(22), columns = ['fr','tr','us'])
print(df)

df2 = pd.DataFrame(data=age, index = range(22), columns = ['boy','kilo','yas'])
print(df2)

cinsiyet = datas.iloc[:,-1].values
print(cinsiyet)

df3 = pd.DataFrame(data = c[:,:1], index = range(22), columns = ['cinsiyet'])
print(df3)


#dataframe birlestirme islemi
s=pd.concat([df,df2], axis=1)
print(s)

s2=pd.concat([s,df3], axis=1)
print(s2)

#egitim ve test
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(s,df3,test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#boy değerleri tablodan alınıyor
boy = s2.iloc[:,3:4].values
print(boy)

#eğitmek için boy kolununun solundaki ve sağındaki değerleri alıyoruz
sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)
x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)

r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)

#Backward Elimination 
import statsmodels.api as sm
#Verileri eleyerek daha iyi bir tahmine ulaşmaya çalışıyoruz
#En yüksek p value alan değişken elenir

#array oluşturuyoruz
X = np.append(arr = np.ones((22,1)).astype(int), values = veri , axis = 1)

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
#OLS raporu çıkartılıyor
r_ols = sm.OLS(endog=boy,exog=X_l)
r = r_ols.fit()
print(r.summary())

#p value yüksek olduğu için 4'ü çıkarıyoruz
X_l = veri.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float)
r_ols = sm.OLS(endog=boy,exog=X_l)
r = r_ols.fit()
print(r.summary())