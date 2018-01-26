import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_absolute_error as mae
import matplotlib.pyplot as plt
import matplotlib.colors as clr


class SGDRegressor_change_predict(SGDRegressor):
    """определения метода, добавляющего новые признаки в тестовую выборку"""
    def predict(self, X):
        if X.shape[1]== 2:
            X = np.array(X)
            size = len(X)
            X = np.hstack((X, np.exp(X[:, 0]).reshape((size, 1))))
            X = np.hstack((X, np.sqrt(np.abs(X[:, 1])).reshape((size, 1))))
            X = np.hstack((X, (X[:, 2] * X[:, 3]).reshape((size, 1))))
        return SGDRegressor.predict(self, X)


X_train, X_test, Y_train, Y_test = utils.load_data(info = True)
    
#добавление новых признаков в обучающую выборку
X_train['exp(a)'] = np.exp(X_train['a'])
X_train['sqrt(b)'] = np.sqrt(abs(X_train['b']))
X_train['exp(a) * sqrt(b)'] = X_train['exp(a)'] * X_train['sqrt(b)']
        
#модель
sgd = SGDRegressor_change_predict(loss = 'epsilon_insensitive', epsilon = 0,
                                  shuffle = False, random_state = 1)

#подбор параметров модели                
parameters_grid = [
    {
    'penalty':['none']
    },
    {
    'penalty':['l1', 'l2'],
    'alpha': np.arange(0, .2, .01)[1:]
    }, 
    {
    'penalty':['elasticnet'],
    'alpha': np.arange(0, .2, .01)[1:],
    'l1_ratio': np.arange(0, .2, .01)[1:], 
    'learning_rate':['constant', 'optimal', 'invscaling']
    }
    ]
grid_cv = GridSearchCV(sgd, parameters_grid, scoring = 'neg_mean_absolute_error', cv = 5, verbose = 1)
grid_cv.fit(X_train, Y_train)


with open('Sgd_model.txt', 'w') as file:
    file.write('модель с лучшими параметрами: \n')
    file.write(str(grid_cv.best_estimator_) + '\n')
    file.write('MAE на тесте: \n')
    file.write(str(mae(grid_cv.best_estimator_.predict(X_test), Y_test)) + '\n')

print('веса: ,', list(zip(X_train.columns,
                          map(lambda x: round(x, 2), grid_cv.best_estimator_.coef_))))

#построение диаграммы рассеивания оценок и ответов
utils.predict_original_plot(X_test, Y_test, grid_cv.best_estimator_)

#построение гистограммы весов
plt.figure(figsize=(10, 5))
plt.bar(np.arange(1, 1 + len(X_train.columns)), grid_cv.best_estimator_.coef_)
plt.xticks(np.arange(1, 1 + len(X_train.columns)), X_train.columns)
plt.show()


