import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from commonTrainTest import *

# Which dataset
datasets = ['a_a', 'a_c', 'b_a', 'b_b', 'b_c', 'c_a', 'c_c', 'd_a', 'd_c']

for subset in datasets:

    dataset = subset
    x_tr = 'x_data_'+dataset+'_train'
    y_tr = 'y_data_'+dataset+'_train'
    x_te = 'x_data_'+dataset+'_test'
    y_te = 'y_data_'+dataset+'_test'

    x_train = locals()[x_tr]
    y_train = locals()[y_tr]
    x_test = locals()[x_te]
    y_test = locals()[y_te]

    cols = ['monthStr', 'PromoInterval', 'StoreType', 'Assortment']
    x_train.drop(cols, axis=1, inplace=True)
    x_test.drop(cols, axis=1, inplace=True)

    cols = list(x_train)
    cols.pop(cols.index('Date'))
    x_train_without_date = x_train.loc[:, cols]
    x_test_without_date = x_test.loc[:, cols]

    rf_model = RandomForestRegressor(
        n_estimators=250, n_jobs=32, verbose=1, random_state=678)

    rf_model.fit(x_train_without_date, y_train)
    rf_predict = rf_model.predict(x_test_without_date)

    rf_rmse = np.sqrt(mean_squared_error(rf_predict, y_test))
    rf_mae = mean_absolute_error(rf_predict, y_test)
    rf_r2 = r2_score(rf_predict, y_test)

    lines = []

    lines.append('Random Forest RMSE: ' + str(rf_rmse))
    lines.append('Random Forest MAE: ' + str(rf_mae))
    lines.append('Random Forest R2 Score: ' + str(rf_r2))
    lines.append('RMSPE: ' + str(rmspe(rf_predict, y_test)))

    with open(dataset+'.txt', 'w') as f:
        f.write('\n'.join(lines))
