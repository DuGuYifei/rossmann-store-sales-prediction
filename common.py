import pandas as pd
import numpy as np
from time import time
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

# load dataset
train = pd.read_csv('./dataset/train.csv', parse_dates=[2])
test = pd.read_csv('./dataset/test.csv', parse_dates=[3])
store = pd.read_csv('./dataset/store.csv')
sample_submission = pd.read_csv('dataset/sample_submission.csv')

features = [
    'Store', 'CompetitionDistance', 'StateHoliday', 'StoreType', 'Assortment',
    'SchoolHoliday', 'Promo', 'Promo2', 'Year', 'Month', 'Day', 'DayOfWeek',
    'WeekOfYear', 'CompetitionOpen', 'PromoOpen', 'IsPromoMonth'
]


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
    data['Date'] = data.Date.astype('datetime64[ns]')
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
    data['CompetitionOpen'] = data.CompetitionOpen.apply(
        lambda x: x if x > 0 else 0)
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + \
        (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
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
                data.loc[(data.monthStr == month) & (
                    data.PromoInterval == interval), 'IsPromoMonth'] = 1
    return data


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
