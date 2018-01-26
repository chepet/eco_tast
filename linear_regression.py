import utils
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
import matplotlib.pyplot as plt

X_train, X_test, Y_train, Y_test = utils.load_data(info = True)

def F_test(X, Y, estimator):
    """проверка гипотезы о том, что константная модель лучше линейной"""
    r2 = r2_score(Y, estimator.predict(X)) ** 2
    F = (r2 / X.shape[1]) / ((1 - r2) / (X.shape[0] - X.shape[1] - 1))
    return F, (X.shape[1], X.shape[0] - X.shape[1] - 1)
    

#модель
lr = LinearRegression()
lr.fit(X_train, Y_train)

with open('linear_regression.txt', 'w') as file:
    file.write('MAE на тесте: \n')
    file.write(str(mae(lr.predict(X_test), Y_test)) + '\n')

print('значения весов: ', list(zip(X_test.columns, lr.coef_))) 
    
#проверка гипотезы
print('значение статистики: ', F_test(X_train, Y_train, lr)[0], '\n',
      'число степеней свободы: ',  F_test(X_train, Y_train, lr)[1])

#построение диаграммы рассеивания оценок и ответов
utils.predict_original_plot(X_test, Y_test, lr)


plt.figure(figsize=(10, 5))
plt.bar(np.arange(1, 1 + len(X_train.columns)), lr.coef_)
plt.xticks(np.arange(1, 1 + len(X_train.columns)), X_train.columns)
plt.show()
