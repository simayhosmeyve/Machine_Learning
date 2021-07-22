#veri kümesinin parçalanıp her parçaya farklı bir karar ağacı kullanılmasıyla çözülür
#tahmin aşamasında çıkan sonuçların ortalaması alınır

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv('salaries.csv')
#print(datas)
grade = datas.iloc[:,1:2]
salaries = datas.iloc[:,2:]

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(random_state=0,n_estimators=10)
#n_estimators kaç karar ağacının çizileceğini seçer
rf_reg.fit(grade.values,salaries.values.ravel())
print(rf_reg.predict([[6.5]]))

plt.scatter(grade,salaries,color="orange")
plt.plot(grade,rf_reg.predict(grade))
plt.show()