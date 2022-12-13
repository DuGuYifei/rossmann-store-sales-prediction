from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from time import time
import numpy as np
import pandas as pd
import xgboost as xgb

# model train params
params = {
	"objective": "reg:linear",
	"booster": "gbtree",
	"eta": 0.3,
	"max_depth": 10,
	"subsample": 0.9,
	"colsample_bytree": 0.7,
	"silent": 1,
	"seed": 1301
}
num_trees = 300

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


def train_predict(dataset, i):
	print("----------" + i + "-------------------------")

	X_train, X_test = train_test_split(dataset, test_size=0.02, random_state=10)
	X_train.reset_index(drop=True, inplace=True)
	X_test.reset_index(drop=True, inplace=True)

	features_create(X_train)
	features_create(X_test)

	dtrain = xgb.DMatrix(X_train[features], np.log1p(X_train.Sales))
	dvalid = xgb.DMatrix(X_test[features], np.log1p(X_test.Sales))
	watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

	print('Training XGBoost model')
	start = time()
	gbm = xgb.train(params, dtrain, num_trees, evals=watchlist,
					early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=50)
	end = time()
	print('Training time: {:2f} s.'.format(end - start))
	# validate
	print('Validating')
	yhat = gbm.predict(dvalid)
	X_test.sort_index(inplace=True)
	error = rmspe(np.expm1(np.log1p(X_test.Sales)), np.expm1(yhat))
	print('RMSPE:{:.6f}'.format(error))
	rf_rmse = np.sqrt(mean_squared_error(np.expm1(yhat), np.log1p(X_test.Sales)))
	rf_mae = mean_absolute_error(np.expm1(yhat), np.log1p(X_test.Sales))
	rf_r2 = r2_score(np.expm1(yhat), np.log1p(X_test.Sales))
	print('RMSE:{:.6f}'.format(rf_rmse))
	print('MAE:{:.6f}'.format(rf_mae))
	print('R2:{:.6f}'.format(rf_r2))
	# store model
	gbm.save_model('../results/xgboost_model' + i + '.json')
	print("----------" + i + "-------------------------")


train = pd.read_csv('../../dataset/train.csv', low_memory=False)
store = pd.read_csv('../../dataset/store.csv', low_memory=False)

store.fillna(0, inplace=True)

train = train.loc[train.Open != 0]
train = train.loc[train.Sales > 0].reset_index(drop=True)
train.drop(['Customers'], axis=1, inplace=True)

train = pd.merge(train, store, on='Store')

# 'StoreType' unique ['c' 'a' 'd' 'b']
# 'Assortment' unique ['a' 'c' 'b']

###data_<StoreType>_<Assortment>

data = train.groupby(['StoreType'])
data_a = data.get_group('a')
data_b = data.get_group('b')
data_c = data.get_group('c')
data_d = data.get_group('d')

data_a = data_a.groupby(['Assortment'])
# unique ['a', 'c']
data_a_a = data_a.get_group('a')
data_a_c = data_a.get_group('c')

data_b = data_b.groupby(['Assortment'])
# unique ['a', 'b', 'c']
data_b_a = data_b.get_group('a')
data_b_b = data_b.get_group('b')
data_b_c = data_b.get_group('c')

data_c = data_c.groupby(['Assortment'])
# unique ['a', 'c']
data_c_a = data_c.get_group('a')
data_c_c = data_c.get_group('c')

data_d = data_d.groupby(['Assortment'])
# unique ['a', 'c']
data_d_a = data_d.get_group('a')
data_d_c = data_d.get_group('c')

train_predict(data_a_a, "data_a_a")
train_predict(data_a_c, "data_a_c")
train_predict(data_b_a, "data_b_a")
train_predict(data_b_b, "data_b_b")
train_predict(data_b_c, "data_b_c")
train_predict(data_c_a, "data_c_a")
train_predict(data_c_c, "data_c_c")
train_predict(data_d_a, "data_d_a")
train_predict(data_d_c, "data_d_c")
