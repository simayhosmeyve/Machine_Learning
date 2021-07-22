import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv('salaries.csv')
#print(datas)
grade = datas.iloc[:,1:2]
print(grade)
salaries = datas.iloc[:,2:]
print(salaries)

#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x = sc.fit_transform(grade)
y = np.ravel(sc.fit_transform(salaries.values.reshape(-1,1)))

from sklearn.svm import SVR
#rbf yerine polinom ya da lineer seçilebilir
svr_reg = SVR(kernel="rbf")
svr_reg.fit(x,y)
plt.scatter(x,y)
plt.plot(x,svr_reg.predict(x))
plt.show()
#örnek tahmin değeri
print(svr_reg.predict([[10]]))