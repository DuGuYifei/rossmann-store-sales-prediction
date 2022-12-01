import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from common import *


train = pd.read_csv('dataset/train.csv', low_memory=False)

store.fillna(0, inplace=True)

train = train.loc[train.Open != 0]
train = train.loc[train.Sales > 0].reset_index(drop=True)

train = pd.merge(train, store, on='Store')

features_create(train)
cols = ['Store', 'monthStr', 'Promo2SinceWeek', 'Promo2SinceYear',
        'PromoInterval', 'CompetitionOpen', 'PromoOpen']
train.drop(cols, axis=1, inplace=True)

train = train.sort_values(by=['Date']).reset_index(drop=True)

X_train = train[train['Date'] < '2015-01-01']
X_test = train[train['Date'] >= '2015-01-01']

cols = list(train)
cols.insert(0, cols.pop(cols.index('Sales')))
cols.pop(cols.index('Date'))
# print(cols)
X_train_without_date = X_train.loc[:, cols]
X_test_without_date = X_test.loc[:, cols]

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(X_train_without_date)
X_train_without_date = scaler.transform(X_train_without_date)
X_test_without_date = scaler.transform(X_test_without_date)

y_train = X_train_without_date[:, 0]
x_train = np.delete(X_train_without_date, 0, 1)
y_test = X_test_without_date[:, 0]
x_test = np.delete(X_test_without_date, 0, 1)

linreg_model = LinearRegression()
linreg_model.fit(x_train, y_train)
linreg_pred = linreg_model.predict(x_test)

linreg_pred = linreg_pred.reshape(-1, 1)
linreg_pred_test_set = np.concatenate([linreg_pred, x_test], axis=1)
linreg_pred_test_set = scaler.inverse_transform(linreg_pred_test_set)

real_test = X_test['Sales'].to_numpy().T

linreg_rmse = np.sqrt(mean_squared_error(
    linreg_pred_test_set[:, 0], real_test))
linreg_mae = mean_absolute_error(linreg_pred_test_set[:, 0], real_test)
linreg_r2 = r2_score(linreg_pred_test_set[:, 0], real_test)
print('Linear Regression RMSE: ', linreg_rmse)
print('Linear Regression MAE: ', linreg_mae)
print('Linear Regression R2 Score: ', linreg_r2)
print('Linear Regression RMSPE: ', rmspe(
    linreg_pred_test_set[:, 0], real_test))


plt.figure(figsize=(15, 7))
plt.plot(train['Date'], train['Sales'])
plt.plot(X_test['Date'], linreg_pred_test_set[:, 0])
plt.title("Sales forecast using Linear Regression")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(["Original Sales", "Predicted Sales"])
# plt.show()
plt.savefig(
    'results/linearRegressionPlots/salesForecastUsingLinearRegressionFullPicture')


train_to_visualise = train[-30:]
test_to_visualise = X_test[-30:]
pred_to_visualise = linreg_pred_test_set[:, 0]
pred_to_visualise = pred_to_visualise[-30:]

plt.clf()
plt.figure(figsize=(15, 7))
plt.plot(np.arange(1, 31), train_to_visualise['Sales'])
plt.plot(np.arange(1, 31), pred_to_visualise)
plt.title("Sales Forecast using Linear Regression")
plt.xlabel("Last 30 records")
plt.ylabel("Sales")
plt.legend(["Original Sales", "Predicted Sales"])
# plt.show()
plt.savefig(
    'results/linearRegressionPlots/salesForecastUsingLinearRegressionLast30Records')
