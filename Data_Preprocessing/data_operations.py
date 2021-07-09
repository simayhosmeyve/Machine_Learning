import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#veri yukleme
datas = pd.read_csv("missing_values.csv")
#print(datas)
#tüm tabloyu gösterir

#veri on isleme
height = datas[['boy']]
print(height)
#tablodaki boy kısmını gösterir

#eksik verilerin veri ortalaması ile doldurulması
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
age = datas.iloc[:,1:4].values
print(age)
imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])
print(age)

#verilerin encoding yapılması
#sayısal verilere dönüştürm

country = datas.iloc[:,0:1].values
print(country)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
country[:,0] = le.fit_transform(datas.iloc[:,0])
print(country)

ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
print(country)

#DataFrame, iki boyutlu veriler ve karşılık gelen etiketleri içeren bir yapıdır
df = pd.DataFrame(data=country,index=range(22),columns=['fr','tr','us'])
print(df)

df2 = pd.DataFrame(data=height,index=range(22),columns=['boy'])
print(df2)

gender= datas.iloc[:,-1].values
print(gender)

df3 = pd.DataFrame(data=gender,index=range(22),columns=['cinsiyet'])
print(df3)

#dataframeleri birleştiriyoruz
#axis{0/’index’, 1/’columns’}, default 0
union = pd.concat([df,df2], axis=1)
print(union)

union2 = pd.concat([union,df3], axis=1)
print(union2)

#veri kümesinin eğitim ve test olarak bölünmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(union,df3,test_size=0.33,random_state=0)
print("Train and Test \n",x_train,"\n",y_train,"\n",x_test,"\n",y_test)

#öznitelik ölçekleme
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

print(X_test)