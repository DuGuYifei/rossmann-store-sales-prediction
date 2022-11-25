import pandas as pd
import numpy as np
import xgboost as xgb
from time import time
from sklearn.model_selection import train_test_split

train = pd.read_csv('../dataset/train.csv', parse_dates=[2])
test = pd.read_csv('../dataset/test.csv', parse_dates=[3])
store = pd.read_csv('../dataset/store.csv')

# change all store in test file to open
test.fillna(1, inplace=True)

# other missing value set to 0
store.fillna(0, inplace=True)

# drop not open and sales small than 0
train = train.loc[train.Open != 0]
train = train.loc[train.Sales > 0].reset_index(drop=True)

# merge data
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')


def features_create(data):
    # change char to num
    mappings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)
    data['StoreType'] = data['StoreType'].astype(int)
    data['Assortment'] = data['Assortment'].astype(int)
    data['StateHoliday'] = data['StateHoliday'].astype(int)

    # time features, use dt to handle
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek
    data['WeekOfYear'] = data.Date.dt.isocalendar().week
    data['WeekOfYear'] = data['WeekOfYear'].astype(int)

    # 'CompetitionOpen'：rival open (last how long time)
    # 'PromoOpen' rival promotion open (last how long time)
    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + (
            data.Month - data.CompetitionOpenSinceMonth)
    data['CompetitionOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)

    # 'IsPromoMonth'：whether in promotion month，1 yes，0 no
    month_to_str = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sept',
                    10: 'Oct',
                    11: 'Nov', 12: 'Dec'}
    data['monthStr'] = data.Month.map(month_to_str)
    data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
    data['IsPromoMonth'] = 0
    for interval in data.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1

    return data


features = ['Store', 'CompetitionDistance', 'StateHoliday', 'StoreType', 'Assortment',
            'SchoolHoliday', 'Promo', 'Promo2',
            'Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear',
            'CompetitionOpen', 'PromoOpen', 'IsPromoMonth']

features_create(train)
features_create(test)

# -----------------------------------------------
# train model
params = {"objective": "reg:linear",
          "booster": "gbtree",
          "eta": 0.3,
          "max_depth": 10,
          "subsample": 0.9,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 1301
          }
num_trees = 300


# evaluation function (sqrt(mean([(yhat - y)/ y)^2]))
# check if y = 0.
def to_weight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1. / (y[ind] ** 2)
    return w


def rmspe(yhat, y):
    w = to_weight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return rmspe


def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    w = to_weight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return "rmspe", rmspe


# Randomly split the training set and test set
X_train, X_test = train_test_split(train, test_size=0.02, random_state=10)

dtrain = xgb.DMatrix(X_train[features], np.log1p(X_train.Sales))
dvalid = xgb.DMatrix(X_test[features], np.log1p(X_test.Sales))

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

# train model
print('Train XGBoost model')
start = time()
gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=50)
end = time()
print('Training time is {:2f} s.'.format(end - start))

# validate
print('Validating')
yhat = gbm.predict(dvalid)
X_test.sort_index(inplace=True)
test.sort_index(inplace=True)
error = rmspe(np.expm1(np.log1p(X_test.Sales)), np.expm1(yhat))
print('RMSPE:{:.6f}'.format(error))

# store model
gbm.save_model('xgboost_model.json')
# -------------------------
# predict file
test_probs = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_ntree_limit)
indices = test_probs < 0
test_probs[indices] = 0
submission = pd.DataFrame({"Id": test["Id"], "Sales": np.expm1(test_probs) * 0.95})
submission.to_csv("xgboost_new_model_1_remove_close.csv", index=False)
