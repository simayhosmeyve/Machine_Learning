#polinomal regresyon
#polinomal ilişkisi olan verilerde
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv('salaries.csv')
#print(datas)
grade = datas.iloc[:,1:2]
print(grade)
salaries = datas.iloc[:,2:]
print(salaries)

'''
#lineer regresyon ile görsel olarak değerlendirme
#doğru değerler gelmezse grade.values olarak değiştirilebilir
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(grade,salaries)

plt.scatter(grade,salaries,color='red')
plt.plot(grade,lr.predict(grade),color='blue')
plt.show()
'''
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#derecesini parametre olarak belirliyoruz
#poly_r = PolynomialFeatures(degree=2)
#4. derece daha başarılı
poly_r = PolynomialFeatures(degree=4)
grade_poly = poly_r.fit_transform(grade)
print(grade_poly) #x0, x, x^2
lin_reg = LinearRegression()
lin_reg.fit(grade_poly,salaries)

plt.scatter(grade,salaries)
plt.plot(grade,lin_reg.predict(poly_r.fit_transform(grade)),color='green')
plt.show()