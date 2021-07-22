import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv('salaries.csv')
#print(datas)
grade = datas.iloc[:,1:2]
print(grade)
salaries = datas.iloc[:,2:]
print(salaries)

from sklearn.tree import DecisionTreeRegressor
reg_dt = DecisionTreeRegressor(random_state=0)
reg_dt.fit(grade.values,salaries.values)
plt.scatter(grade.values,salaries.values,color="orange")
plt.plot(grade.values,reg_dt.predict(grade.values))
plt.show()