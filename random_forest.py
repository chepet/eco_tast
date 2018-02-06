import utils
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 

X_train, X_test, Y_train, Y_test = utils.load_data(info = True)

rf = RandomForestRegressor(criterion = 'mae', random_state = 1)

#подбор параметров модели                
parameters_grid = {
    'n_estimators': list(range(10, 100, 10)),
    'max_depth': list(range(5, 25, 5))
    }

grid_cv = GridSearchCV(rf, parameters_grid, scoring = 'neg_mean_absolute_error', cv = 5, verbose = 1)
grid_cv.fit(X_train, Y_train)

with open('random_forest.txt', 'w') as file:
    file.write('модель с лучшими параметрами: \n')
    file.write(str(grid_cv.best_estimator_) + '\n')
    file.write('MAE на тесте: \n')
    file.write(str(mean_absolute_error(grid_cv.best_estimator_.predict(X_test),
                                       Y_test)) + '\n')


#построение диаграммы рассеивания оценок и ответов
utils.predict_original_plot(X_test, Y_test, grid_cv.best_estimator_)

#композиция
pr = []
for i in range(100):
    grid_cv.best_estimator_.random_state = i
    grid_cv.best_estimator_.fit(X_train, Y_train)
    pr.append(mean_absolute_error(grid_cv.best_estimator_.predict(X_test),
                                       Y_test))

print(sum(pr) / len(pr))
print(grid_cv.best_estimator_)
    
