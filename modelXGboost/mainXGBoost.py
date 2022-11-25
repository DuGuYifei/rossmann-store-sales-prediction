import xgboost as xgb
from sklearn.model_selection import train_test_split
from common import *

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

features_create(train)
features_create(test)

# Training the model
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

# Randomly split the training set and test set
X_train, X_test = train_test_split(train, test_size=0.02, random_state=10)

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
test.sort_index(inplace=True)
error = rmspe(np.expm1(np.log1p(X_test.Sales)), np.expm1(yhat))
print('RMSPE:{:.6f}'.format(error))

# store model
gbm.save_model('../results/xgboost_model.json')
# -------------------------
# predict file
test_probs = gbm.predict(
    xgb.DMatrix(
        test[features]
    ),
    ntree_limit=gbm.best_ntree_limit
)
indices = test_probs < 0
test_probs[indices] = 0
submission = pd.DataFrame({
    "Id": test["Id"],
    "Sales": np.expm1(test_probs) * 0.95
})
submission.to_csv(
    "../results/XGBoost/xgboost_new_model_1_remove_close.csv", index=False)
