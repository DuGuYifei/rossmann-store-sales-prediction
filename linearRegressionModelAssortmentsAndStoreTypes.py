from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from commonTrainTest import *


def linearRegression(data_a_a):
    X_train, X_test = train_test_split(data_a_a, test_size=0.02, random_state=10)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    features_create(X_train)
    features_create(X_test)

    cols = ['Store', 'monthStr', 'Promo2SinceWeek', 'Promo2SinceYear',
            'PromoInterval', 'CompetitionOpen', 'PromoOpen']
    X_train.drop(cols, axis=1, inplace=True)
    X_test.drop(cols, axis=1, inplace=True)

    cols = list(X_train)
    cols.insert(0, cols.pop(cols.index('Sales')))
    cols.pop(cols.index('Date'))
    # print(cols)
    X_train_without_date = X_train.loc[:, cols]
    X_test_without_date = X_test.loc[:, cols]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_train_without_date)
    X_train_without_date = scaler.transform(X_train_without_date)
    X_test_without_date = scaler.transform(X_test_without_date)

    y_train = X_train_without_date[:, 0]
    x_train = np.delete(X_train_without_date, 0, 1)
    y_test = X_test_without_date[:, 0]
    x_test = np.delete(X_test_without_date, 0, 1)

    linreg_model = LinearRegression()
    linreg_model.fit(x_train, y_train)
    linreg_pred = linreg_model.predict(x_test)

    linreg_pred = linreg_pred.reshape(-1, 1)
    linreg_pred_test_set = np.concatenate([linreg_pred, x_test], axis=1)
    linreg_pred_test_set = scaler.inverse_transform(linreg_pred_test_set)

    real_test = X_test['Sales'].to_numpy().T

    linreg_rmse = np.sqrt(mean_squared_error(
        linreg_pred_test_set[:, 0], real_test))
    linreg_mae = mean_absolute_error(linreg_pred_test_set[:, 0], real_test)
    linreg_r2 = r2_score(linreg_pred_test_set[:, 0], real_test)
    rmspe_result = rmspe(linreg_pred_test_set[:, 0], real_test)
    print('Linear Regression RMSE: ', round(linreg_rmse,6))
    print('Linear Regression MAE: ', round(linreg_mae,6))
    print('Linear Regression R2 Score: ', round(linreg_r2,6))
    print('Linear Regression RMSPE: ', round(rmspe_result,6))

print("data_a_a")
linearRegression(data_a_a)
print("\ndata_a_c")
linearRegression(data_a_c)
print("\ndata_b_a")
linearRegression(data_b_a)
print("\ndata_b_b")
linearRegression(data_b_b)
print("\ndata_b_c")
linearRegression(data_b_c)
print("\ndata_c_a")
linearRegression(data_c_a)
print("\ndata_c_c")
linearRegression(data_c_c)
print("\ndata_d_a")
linearRegression(data_d_a)
print("\ndata_d_c")
linearRegression(data_d_c)