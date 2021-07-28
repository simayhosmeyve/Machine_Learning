import pandas as pd

datas = pd.read_csv('salaries.csv')
grade = datas.iloc[:,1:2]
salaries = datas.iloc[:,2:]

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(random_state=0,n_estimators=10)
rf_reg.fit(grade.values,salaries.values.ravel())

from sklearn.metrics import r2_score
score = r2_score(salaries.values,rf_reg.predict(grade.values))
print(score)

#tahminlerin gerçek değerlerden uzaklaşması durumunda r2 skorunun azaldığını gözlemleriz
k = grade.values + 0.5
score2 = r2_score(salaries.values,rf_reg.predict(k))
print(score2)