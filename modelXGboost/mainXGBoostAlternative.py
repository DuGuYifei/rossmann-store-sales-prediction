import mainXGBoostAlternative as xgb
from sklearn.model_selection import train_test_split
from common import *

# change all store in test file to open
test.fillna(1, inplace=True)

# other missing value set to 0
store.fillna(0, inplace=True)

# merge data
train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

features_create(train)
features_create(test)

# XGBoost
# parameter adjust
# eta：iteration step length
# max_depth：The maximum depth of a single regression tree, smaller results in underfitting, larger results in overfitting;
# subsample：Between 0-1, control the random sampling ratio of each tree, reduce the value of this parameter, the algorithm will be more conservative and avoid overfitting. But if this value is set too small, it may lead to underfitting;
# colsample_bytree：Between 0-1, used to control the proportion of each randomly sampled feature;
# num_trees：Iteration steps count.

# params = {'objective': 'reg:linear',
#		   'eta': 0.03,
#		   'max_depth': 11,
#		   'subsample': 0.5,
#		   'colsample_bytree': 0.5,
#		   'silent': 1,
#		   'seed': 10
#		   }
# num_trees = 200

# params = {'objective': 'reg:linear',
#		   'eta': 0.1,
#		   'max_depth': 10,
#		   'subsample': 0.8,
#		   'colsample_bytree': 0.5,
#		   'silent': 1,
#		   'seed': 997
#		   }
# num_trees = 150

# params = {"objective": "reg:linear",
#		   "booster" : "gbtree",
#		   "eta": 0.1,
#		   "max_depth": 10,
#		   "subsample": 0.85,
#		   "colsample_bytree": 0.4,
#		   "min_child_weight": 6,
#		   "silent": 1,
#		   "thread": 1,
#		   "seed": 1301
#		   }
# num_trees = 100

params = {'objective': 'reg:linear',
		  'eta': 0.03,
		  'max_depth': 11,
		  'subsample': 0.8,
		  'colsample_bytree': 0.5,
		  'silent': 1,
		  'seed': 1301
		  }
num_trees = 200


def rmspe_xg(yhat, y):
	y = y.get_label()
	y = np.expm1(y)
	yhat = np.expm1(yhat)
	w = to_weight(y)
	rmspe = np.sqrt(np.mean(w * (y-yhat)**2))
	return "rmspe", rmspe


def to_weight(y):
	w = np.zeros(y.shape, dtype=float)
	ind = y != 0
	w[ind] = 1./(y[ind]**2)
	return w


def rmspe(yhat, y):
	w = to_weight(y)
	rmspe = np.sqrt(np.mean(w * (y-yhat)**2))
	return rmspe


def objective(trial, train=train):
	params = {
		"lambda": trial.suggest_loguniform('lambda', 0.01, 10.0),
		"alpha": trial.suggest_loguniform('alpha', 0.01, 10.0),
		"colsample_bytree": trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
		"subsample": trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
		"learning_rate": trial.suggest_categorical('learning_rate',
												   [0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
		"n_estimators": 10,
		"max_depth": trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13]),
		"random_state": trial.suggest_categorical('random_state', [24, 48, 2020]),
		"min_child_weight": trial.suggest_int('min_child_weight', 1, 300),
	}

	X_train, X_test = train_test_split(train, test_size=0.2, random_state=2)

	dtrain = xgb.DMatrix(X_train[features], np.log1p(X_train.Sales))
	dvalid = xgb.DMatrix(X_test[features], np.log1p(X_test.Sales))
	dtest = xgb.DMatrix(test[features])

	watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
	gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, feval=rmspe_xg,
					verbose_eval=False)

	yhat = gbm.predict(dvalid, ntree_limit=gbm.best_ntree_limit)
	yhat = np.expm1(yhat)

	y = X_test.Sales
	w = to_weight(y)
	rmspe_ = np.sqrt(np.mean(w * (y - yhat) ** 2))

	return rmspe_


X_train, X_test = train_test_split(train, test_size=0.2, random_state=2)

dtrain = xgb.DMatrix(X_train[features], np.log1p(X_train.Sales))
dvalid = xgb.DMatrix(X_test[features], np.log1p(X_test.Sales))
dtest = xgb.DMatrix(test[features])

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

# Train model
print('Training XGBoost model')
start = time()
gbm = xgb.train(params, dtrain, num_trees, evals=watchlist,
				early_stopping_rounds=50, feval=rmspe_xg, verbose_eval=50)
end = time()
print('Training time is {:2f} s.'.format(end-start))

print('validating')
X_test.sort_index(inplace=True)
test.sort_index(inplace=True)
yhat = gbm.predict(dvalid)
error = rmspe(np.expm1(np.log1p(X_test.Sales)), np.expm1(yhat))

print('RMSPE:{:.6f}'.format(error))


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print('Number of finished trials:', len(study.trials))
print('best trial:', study.best_trial.params)

# Submit file
test_probs = gbm.predict(xgb.DMatrix(
	X_test[features]), ntree_limit=gbm.best_ntree_limit)
indices = test_probs < 0
test_probs[indices] = 0
submission = pd.DataFrame({
	"Id": test["Id"],
	"Sales": np.expm1(test_probs)*0.95,
})

submission.to_csv("./allAdjustments/xgboost.csv", index=False)
