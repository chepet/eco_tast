import seaborn
import pylab
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
seaborn.set()

#построенеи гистограммы
def hist(data):
    data.hist(figsize = (10,5))
    pylab.suptitle(u'гистограммы')

    
#построение корреляционной таблицы   
def corr_pearson(data):
    plt.figure()
    seaborn.heatmap(data.corr(method = 'pearson'), square=True)
    plt.title(u'корреляция пирсона')
    plt.legend()


#построение данных в R^3
def plot_3d(data):
    fig = pylab.figure()
    axes = Axes3D(fig)
    a = np.array(data['a'])
    b = np.array(data['b'])
    y = np.array(data['y'])
    axes.scatter3D(a, b, y)
    axes.set_xlabel('a')
    axes.set_ylabel('b')
    axes.set_zlabel('y')
    axes.set_title(u'Поверхность зависимости "y" от "b" и "a"')
    plt.legend()
    pylab.show()

#построенеи таблицы парных отношений 
def pairplot(data):
    seaborn.pairplot(data)
    pylab.suptitle(u'Парные отношения в данных')
    plt.savefig('pair.png')
    plt.show()

#отрисовака всех графиков
def all_plots(data):
    hist(data)
    corr_pearson(data)
    plot_3d(data)
    pairplot(data)


data = pd.read_csv('nonlinear_rg.csv', index_col = 'date')

print('первые 5 столбцов исходных данных:')
print(data.head())
print('послдение 5 столбцов исходных данных:')
print(data.tail())
print('размер исходной выборки:', data.shape)

#проверка наличия пропусков
if len(np.unique(data.isnull())) == 1:
    print('пропуски отсутствуют')
else:
    print('проспуски в столбцах:')
    print([(name, sum(data[name].isnull() == True)) for name in data.columns
           if True in data[name].isnull().unique()])

print('краткий обзор данных:')
print(data.info())
print('общие статистические данные:')
print(data.describe())
print('значение статистики и p-value:')
print(scipy.stats.pearsonr(data['a'], data['y']))




all_plots(data)

