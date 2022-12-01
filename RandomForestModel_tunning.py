import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from common import *


train = pd.read_csv('dataset/train.csv', low_memory=False)
store = pd.read_csv('dataset/store.csv', low_memory=False)

store.fillna(0, inplace=True)

train = train.loc[train.Open != 0]
train = train.loc[train.Sales > 0].reset_index(drop=True)
train.drop(['Customers'], axis=1, inplace=True)

train = pd.merge(train, store, on='Store')

features_create(train)
cols = ['monthStr', 'PromoInterval']
train.drop(cols, axis=1, inplace=True)

train = train.sort_values(by=['Date']).reset_index(drop=True)
X_train = train[train['Date'] < '2015-01-01']
X_test = train[train['Date'] >= '2015-01-01']

cols = list(train)
cols.insert(0, cols.pop(cols.index('Sales')))
cols.pop(cols.index('Date'))
X_train_without_date = X_train.loc[:, cols]
X_test_without_date = X_test.loc[:, cols]

y_train = X_train_without_date['Sales']
x_train = X_train_without_date.drop(['Sales'], axis=1)
y_test = X_test_without_date['Sales']
x_test = X_test_without_date.drop(['Sales'], axis=1)

rf_model = RandomForestRegressor(
    n_estimators=200, oob_score=True, n_jobs=32, verbose=1, random_state=678)

rf_model.fit(x_train, y_train)
rf_predict = rf_model.predict(x_test)

rf_rmse = np.sqrt(mean_squared_error(rf_predict, y_test))
rf_mae = mean_absolute_error(rf_predict, y_test)
rf_r2 = r2_score(rf_predict, y_test)
print('oob score :', rf_model.oob_score_)
print('Random Forest RMSE: ', rf_rmse)
print('Random Forest MAE: ', rf_mae)
print('Random Forest R2 Score: ', rf_r2)
print('RMSPE: ', rmspe(rf_predict, y_test))

pd.options.display.float_format = '{:.5f}'.format
important_features = pd.DataFrame(
    rf_model.feature_importances_, index=x_train.columns)
print(important_features.sort_values(by=0, ascending=False))

predict_df = pd.DataFrame(rf_predict)
predict_df.columns = ['Predicted_sales']
predict_df = pd.concat([X_test.reset_index(drop=True),
                       predict_df], axis=1, sort=False)
train['Date'] = train['Date'].apply(lambda dt: dt.replace(day=1))
predict_df['Date'] = predict_df['Date'].apply(lambda dt: dt.replace(day=1))
train = train.groupby(['Date'])['Sales'].sum().reset_index()
predict_df = predict_df.groupby(
    ['Date'])['Predicted_sales'].sum().reset_index()

plt.figure(figsize=(15, 7))
plt.plot(train['Date'], train['Sales'])
plt.plot(predict_df['Date'], predict_df['Predicted_sales'])
plt.title("Sales Forecast test using Random Forest")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(["Original Sales", "Predicted Sales"])
plt.show()
