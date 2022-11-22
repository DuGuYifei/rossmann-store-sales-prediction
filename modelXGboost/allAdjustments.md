# RESULT

* eta：iteration step length
* max_depth：The maximum depth of a single regression tree, smaller results in underfitting, larger results in overfitting;
* subsample：Between 0-1, control the random sampling ratio of each tree, reduce the value of this parameter, the algorithm will be more conservative and avoid overfitting. But if this value is set too small, it may lead to underfitting;
* colsample_bytree：Between 0-1, used to control the proportion of each randomly sampled feature;
* num_trees：Iteration steps count.

## Eighth trial - remove closed stores and stores without sales rows of seventh trial
Test result: 0.13614

## Seventh trial
Parameters
```
params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.3,
          "max_depth": 10,
          "subsample": 0.9,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 1301
          }
num_trees = 300
```
Test result: 0.18936

## Sixth trial - remove auto optimization of fourth trial
Test result: 0.29979

## Fifth trial - remove auto optimization of third trial
Test result: 0.26503

## Fourth trial
Parameters
```
params = {'objective': 'reg:linear',
          'eta': 0.03,
          'max_depth': 11,
          'subsample': 0.8,
          'colsample_bytree': 0.5,
          'silent': 1,
          'seed': 1301
          }
num_trees = 200
```
Test result: 0.29979

Best is trial 17 with value: 0.29723525115684474.

Output:
```
Train a XGBoost model
[12:21:29] WARNING: ../src/objective/regression_obj.cu:203: reg:linear is now deprecated in favor of reg:squarederror.
[12:21:29] WARNING: ../src/learner.cc:627: 
Parameters: { "silent" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[0]	train-rmse:7.34416	train-rmspe:0.91061	eval-rmse:7.35833	eval-rmspe:0.91213
[50]	train-rmse:2.02298	train-rmspe:0.78714	eval-rmse:2.02821	eval-rmspe:0.78844
[100]	train-rmse:0.79496	train-rmspe:0.41841	eval-rmse:0.80338	eval-rmspe:0.41182
[150]	train-rmse:0.52294	train-rmspe:0.28707	eval-rmse:0.53754	eval-rmspe:0.26622
[199]	train-rmse:0.42818	train-rmspe:0.26351	eval-rmse:0.44839	eval-rmspe:0.23393
Training time is 216.868849 s.
validating
[I 2022-11-22 12:25:07,430] A new study created in memory with name: no-name-ba41605b-725a-4c3a-b97f-6f1ee9b1bc91
RMSPE:496273.821855
/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:48: FutureWarning: suggest_loguniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use :func:`~optuna.trial.Trial.suggest_float` instead.
/opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:49: FutureWarning: suggest_loguniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use :func:`~optuna.trial.Trial.suggest_float` instead.
[12:25:08] WARNING: ../src/learner.cc:627: 
Parameters: { "n_estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


/opt/conda/lib/python3.7/site-packages/xgboost/core.py:94: UserWarning: ntree_limit is deprecated, use `iteration_range` or model slicing instead.
  UserWarning
[I 2022-11-22 12:26:34,130] Trial 0 finished with value: 0.6434057844488567 and parameters: {'lambda': 0.09109386345020429, 'alpha': 6.566249699222988, 'colsample_bytree': 0.3, 'subsample': 0.4, 'learning_rate': 0.012, 'max_depth': 5, 'random_state': 48, 'min_child_weight': 170}. Best is trial 0 with value: 0.6434057844488567.
[12:26:35] WARNING: ../src/learner.cc:627: 
Parameters: { "n_estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[I 2022-11-22 12:28:24,838] Trial 1 finished with value: 0.8086857495843596 and parameters: {'lambda': 0.6161472182767258, 'alpha': 0.09931225116805398, 'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.008, 'max_depth': 7, 'random_state': 2020, 'min_child_weight': 281}. Best is trial 0 with value: 0.6434057844488567.
[12:28:25] WARNING: ../src/learner.cc:627: 
Parameters: { "n_estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[I 2022-11-22 12:31:49,876] Trial 2 finished with value: 0.33411894651109886 and parameters: {'lambda': 0.010452987587472212, 'alpha': 0.3492741749380972, 'colsample_bytree': 0.5, 'subsample': 0.8, 'learning_rate': 0.018, 'max_depth': 11, 'random_state': 24, 'min_child_weight': 196}. Best is trial 2 with value: 0.33411894651109886.
[12:31:51] WARNING: ../src/learner.cc:627: 
Parameters: { "n_estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[I 2022-11-22 12:36:24,188] Trial 3 finished with value: 0.6369592008256009 and parameters: {'lambda': 5.955807315184146, 'alpha': 0.031166054340154976, 'colsample_bytree': 0.7, 'subsample': 0.8, 'learning_rate': 0.01, 'max_depth': 13, 'random_state': 24, 'min_child_weight': 158}. Best is trial 2 with value: 0.33411894651109886.
[12:36:25] WARNING: ../src/learner.cc:627: 
Parameters: { "n_estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[I 2022-11-22 12:38:37,256] Trial 4 finished with value: 0.7644400475438599 and parameters: {'lambda': 0.01957562238280073, 'alpha': 0.025872580731004773, 'colsample_bytree': 0.3, 'subsample': 0.8, 'learning_rate': 0.009, 'max_depth': 9, 'random_state': 48, 'min_child_weight': 68}. Best is trial 2 with value: 0.33411894651109886.
[12:38:38] WARNING: ../src/learner.cc:627: 
Parameters: { "n_estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[I 2022-11-22 12:41:42,261] Trial 5 finished with value: 0.30218705732798823 and parameters: {'lambda': 0.04215542766494045, 'alpha': 3.737023205299593, 'colsample_bytree': 1.0, 'subsample': 0.8, 'learning_rate': 0.02, 'max_depth': 7, 'random_state': 24, 'min_child_weight': 130}. Best is trial 5 with value: 0.30218705732798823.
[12:41:43] WARNING: ../src/learner.cc:627: 
Parameters: { "n_estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[I 2022-11-22 12:43:30,393] Trial 6 finished with value: 0.7326610766536766 and parameters: {'lambda': 0.5109119121364323, 'alpha': 0.06516632353612481, 'colsample_bytree': 0.3, 'subsample': 0.9, 'learning_rate': 0.01, 'max_depth': 7, 'random_state': 2020, 'min_child_weight': 39}. Best is trial 5 with value: 0.30218705732798823.
[12:43:31] WARNING: ../src/learner.cc:627: 
Parameters: { "n_estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[I 2022-11-22 12:45:45,645] Trial 7 finished with value: 0.6345498664881773 and parameters: {'lambda': 4.852952001616408, 'alpha': 0.2531167568310673, 'colsample_bytree': 0.3, 'subsample': 0.6, 'learning_rate': 0.012, 'max_depth': 9, 'random_state': 48, 'min_child_weight': 159}. Best is trial 5 with value: 0.30218705732798823.
[12:45:46] WARNING: ../src/learner.cc:627: 
Parameters: { "n_estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[I 2022-11-22 12:48:49,900] Trial 8 finished with value: 0.7742722261192538 and parameters: {'lambda': 0.057865672352490044, 'alpha': 8.54473507236158, 'colsample_bytree': 0.5, 'subsample': 1.0, 'learning_rate': 0.008, 'max_depth': 11, 'random_state': 24, 'min_child_weight': 69}. Best is trial 5 with value: 0.30218705732798823.
[12:48:51] WARNING: ../src/learner.cc:627: 
Parameters: { "n_estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[I 2022-11-22 12:53:02,663] Trial 9 finished with value: 0.6104994878157195 and parameters: {'lambda': 0.5293824770984632, 'alpha': 0.013872178694848374, 'colsample_bytree': 1.0, 'subsample': 0.7, 'learning_rate': 0.01, 'max_depth': 9, 'random_state': 48, 'min_child_weight': 135}. Best is trial 5 with value: 0.30218705732798823.
[12:53:03] WARNING: ../src/learner.cc:627: 
Parameters: { "n_estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[I 2022-11-22 12:55:33,326] Trial 10 finished with value: 0.32287214450174817 and parameters: {'lambda': 0.10577601711441614, 'alpha': 1.2890622874752578, 'colsample_bytree': 0.6, 'subsample': 0.5, 'learning_rate': 0.02, 'max_depth': 7, 'random_state': 24, 'min_child_weight': 240}. Best is trial 5 with value: 0.30218705732798823.
[12:55:34] WARNING: ../src/learner.cc:627: 
Parameters: { "n_estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[I 2022-11-22 12:58:03,490] Trial 11 finished with value: 0.3232549558558385 and parameters: {'lambda': 0.11405416532196157, 'alpha': 1.849596257407785, 'colsample_bytree': 0.6, 'subsample': 0.5, 'learning_rate': 0.02, 'max_depth': 7, 'random_state': 24, 'min_child_weight': 258}. Best is trial 5 with value: 0.30218705732798823.
[12:58:04] WARNING: ../src/learner.cc:627: 
Parameters: { "n_estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[I 2022-11-22 13:01:21,427] Trial 12 finished with value: 0.3028998843312057 and parameters: {'lambda': 0.038214474457997874, 'alpha': 1.4766273082664854, 'colsample_bytree': 1.0, 'subsample': 0.5, 'learning_rate': 0.02, 'max_depth': 7, 'random_state': 24, 'min_child_weight': 227}. Best is trial 5 with value: 0.30218705732798823.
[13:01:22] WARNING: ../src/learner.cc:627: 
Parameters: { "n_estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[I 2022-11-22 13:04:35,957] Trial 13 finished with value: 0.30354673760350787 and parameters: {'lambda': 0.030132510809431565, 'alpha': 1.933807450299203, 'colsample_bytree': 1.0, 'subsample': 0.5, 'learning_rate': 0.02, 'max_depth': 7, 'random_state': 24, 'min_child_weight': 215}. Best is trial 5 with value: 0.30218705732798823.
[13:04:37] WARNING: ../src/learner.cc:627: 
Parameters: { "n_estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[I 2022-11-22 13:06:50,553] Trial 14 finished with value: 0.40237523053701646 and parameters: {'lambda': 0.26515813598515675, 'alpha': 0.7801308783370847, 'colsample_bytree': 1.0, 'subsample': 0.9, 'learning_rate': 0.014, 'max_depth': 5, 'random_state': 24, 'min_child_weight': 110}. Best is trial 5 with value: 0.30218705732798823.
[13:06:51] WARNING: ../src/learner.cc:627: 
Parameters: { "n_estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[I 2022-11-22 13:10:16,738] Trial 15 finished with value: 0.41581509062442057 and parameters: {'lambda': 0.034843633412974744, 'alpha': 3.372704455480161, 'colsample_bytree': 0.4, 'subsample': 0.7, 'learning_rate': 0.016, 'max_depth': 13, 'random_state': 24, 'min_child_weight': 100}. Best is trial 5 with value: 0.30218705732798823.
[13:10:17] WARNING: ../src/learner.cc:627: 
Parameters: { "n_estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[I 2022-11-22 13:13:28,007] Trial 16 finished with value: 0.30004798667932064 and parameters: {'lambda': 1.6412468885483682, 'alpha': 0.6273817408301428, 'colsample_bytree': 0.9, 'subsample': 0.6, 'learning_rate': 0.02, 'max_depth': 7, 'random_state': 2020, 'min_child_weight': 299}. Best is trial 16 with value: 0.30004798667932064.
[13:13:29] WARNING: ../src/learner.cc:627: 
Parameters: { "n_estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[I 2022-11-22 13:16:44,524] Trial 17 finished with value: 0.29723525115684474 and parameters: {'lambda': 1.7731887897665901, 'alpha': 0.6600414243859755, 'colsample_bytree': 0.9, 'subsample': 0.6, 'learning_rate': 0.02, 'max_depth': 7, 'random_state': 2020, 'min_child_weight': 10}. Best is trial 17 with value: 0.29723525115684474.
[13:16:45] WARNING: ../src/learner.cc:627: 
Parameters: { "n_estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[I 2022-11-22 13:20:00,525] Trial 18 finished with value: 0.33923218600563626 and parameters: {'lambda': 2.118670730125683, 'alpha': 0.46254023948109835, 'colsample_bytree': 0.9, 'subsample': 0.6, 'learning_rate': 0.016, 'max_depth': 7, 'random_state': 2020, 'min_child_weight': 7}. Best is trial 17 with value: 0.29723525115684474.
[13:20:01] WARNING: ../src/learner.cc:627: 
Parameters: { "n_estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


[I 2022-11-22 13:25:44,584] Trial 19 finished with value: 0.3892413894057763 and parameters: {'lambda': 1.7169090133365248, 'alpha': 0.1601151020750068, 'colsample_bytree': 0.9, 'subsample': 0.6, 'learning_rate': 0.014, 'max_depth': 13, 'random_state': 2020, 'min_child_weight': 297}. Best is trial 17 with value: 0.29723525115684474.
Number of finished trials: 20
best trial: {'lambda': 1.7731887897665901, 'alpha': 0.6600414243859755, 'colsample_bytree': 0.9, 'subsample': 0.6, 'learning_rate': 0.02, 'max_depth': 7, 'random_state': 2020, 'min_child_weight': 10}
```

## Third trial
Parameters
```
params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.1,
          "max_depth": 10,
          "subsample": 0.85,
          "colsample_bytree": 0.4,
          "min_child_weight": 6,
          "silent": 1,
          "thread": 1,
          "seed": 1301
          }
num_trees = 100
```
Best is trial 15 with value: 0.6202222015773905.



## Second trial
Parameters
```
params = {'objective': 'reg:linear',
          'eta': 0.1,
          'max_depth': 10,
          'subsample': 0.8,
          'colsample_bytree': 0.5,
          'silent': 1,
          'seed': 997
          }
num_trees = 100
```

best trial: 0.3....

## First trial
Parameters
```
params = {'objective': 'reg:linear',
          'eta': 0.03,
          'max_depth': 11,
          'subsample': 0.5,
          'colsample_bytree': 0.5,
          'silent': 1,
          'seed': 10
          }
num_trees = 200
```

Output: Best is trial 1 with value: 0.26345339310859456.

Test result: 0.30063