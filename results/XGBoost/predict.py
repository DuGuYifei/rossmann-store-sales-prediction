from common import *
import sys
sys.path.insert(0, '../../')
import xgboost as xgb

test.fillna(1, inplace=True)
store.fillna(0, inplace=True)
test = pd.merge(test, store, on='Store')

features_create(test)

# load model and predict
model_xgb = xgb.Booster()
model_xgb.load_model("xgboost_model.json")
test_probs = model_xgb.predict(xgb.DMatrix(
	test[features]), ntree_limit=model_xgb.best_ntree_limit)

# adjust the format to output as csv which only contains column id and column predict sales
indices = test_probs < 0
test_probs[indices] = 0
submission = pd.DataFrame(
	{"Id": test["Id"], "Sales": np.expm1(test_probs)*0.95})
submission.to_csv("predict.csv", index=False)
