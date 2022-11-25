import xgboost as xgb
import pandas as pd
import numpy as np

test = pd.read_csv('../../dataset/test.csv', parse_dates=[3])
store = pd.read_csv('../../dataset/store.csv')
test.fillna(1, inplace=True)
store.fillna(0, inplace=True)
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

features_create(test)

# load model and predict
model_xgb = xgb.Booster()
model_xgb.load_model("xgboost_model.json")
test_probs = model_xgb.predict(xgb.DMatrix(test[features]), ntree_limit=model_xgb.best_ntree_limit)

# adjust the format to output as csv which only contains column id and column predict sales
indices = test_probs < 0
test_probs[indices] = 0
submission = pd.DataFrame({"Id": test["Id"], "Sales": np.expm1(test_probs)*0.95})
submission.to_csv("predict.csv", index=False)
