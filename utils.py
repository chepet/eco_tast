import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(info = False):
    """фукнция загрузки данных"""
    try:
        data = pd.read_csv('nonlinear_rg.csv', index_col = 'date')
        X_test = data[-120:].drop('y', axis = 1)
        X_train = data[:-120].drop('y', axis = 1)
        Y_test = data['y'][-120:]
        Y_train = data['y'][:-120]
        if info:
            print('размер исходной таблицы данных:', data.shape)
            print('размер обучающей выборки:', X_train.shape)
            print('размер тестовой выборки:', X_test.shape)
        return  X_train, X_test, Y_train, Y_test
    except FileNotFoundError:
        print('файл отсутствует')
                         
def predict_original_plot(X_test, Y_test, estimator):
    """построение диаграммы рассеивания оценок и ответов"""
    x = np.linspace(min(estimator.predict(X_test)),
                    max(estimator.predict(X_test)), 50)
    plt.scatter(estimator.predict(X_test), Y_test, color = 'blue')
    plt.title("""диаграмма рассеивания предсказанного и
          точного значения""")
    plt.plot(x, x, color = 'black')
    plt.xlabel('предсказанное значение')
    plt.ylabel('фактическое значние')
    plt.grid()
    plt.show()
