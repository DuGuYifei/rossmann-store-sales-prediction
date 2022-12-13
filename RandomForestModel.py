import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from common import *

train = pd.read_csv('dataset/train.csv', low_memory=False)
test = pd.read_csv('dataset/test.csv', low_memory=False)
store = pd.read_csv('dataset/store.csv', low_memory=False)

test.fillna(1, inplace=True)
store.fillna(0, inplace=True)

train = train.loc[train.Open != 0]
train = train.loc[train.Sales > 0].reset_index(drop=True)
train.drop(['Customers'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

features_create(train)
features_create(test)

cols = ['monthStr', 'PromoInterval']
train.drop(cols, axis=1, inplace=True)
test.drop(cols, axis=1, inplace=True)

cols = list(train)
cols.insert(0, cols.pop(cols.index('Sales')))
cols.pop(cols.index('Date'))
X_train_without_date = train.loc[:, cols]

y_train = X_train_without_date['Sales']
x_train = X_train_without_date.drop(['Sales'], axis=1)

rf_model = RandomForestRegressor(
	n_estimators=200, oob_score=True, n_jobs=32, verbose=1, random_state=678)
rf_model.fit(x_train, y_train)

print('oob score :', rf_model.oob_score_)

cols.pop(cols.index('Sales'))
test_without_date = test.loc[:, cols]
rf_predict = rf_model.predict(test_without_date)

predict_df = pd.DataFrame(rf_predict)
predict_df.columns = ['Predicted_sales']
predict_df = pd.concat([test, predict_df], axis=1, sort=False)
predict_df = predict_df.groupby(
	['Date'])['Predicted_sales'].sum().reset_index()

plt.figure(figsize=(15, 7))
plt.plot(predict_df['Date'], predict_df['Predicted_sales'])
plt.title("Sales Forecast using Random Forest")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()
