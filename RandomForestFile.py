import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import date, timedelta, datetime
from common import *


def random_forest_to_file(file_name='random_forest', n_estimators=200, n_jobs=32, verbose=1, random_state=678):
    train = pd.read_csv('dataset/train.csv', low_memory=False)
    store = pd.read_csv('dataset/store.csv', low_memory=False)

    store.fillna(0, inplace=True)

    train = train.loc[train.Open != 0]
    train = train.loc[train.Sales > 0].reset_index(drop=True)
    # train.drop(['Customers'], axis=1, inplace=True)

    train = pd.merge(train, store, on='Store')

    features_create(train)
    cols = ['monthStr', 'PromoInterval']
    train.drop(cols, axis=1, inplace=True)

    train = train.sort_values(by=['Date']).reset_index(drop=True)
    X_train = train

    cols = list(train)
    cols.insert(0, cols.pop(cols.index('Sales')))
    cols.pop(cols.index('Date'))
    X_train_without_date = X_train.loc[:, cols]

    y_train = X_train_without_date['Sales']
    x_train = X_train_without_date.drop(['Sales'], axis=1)

    rf_model = RandomForestRegressor(
        n_estimators=n_estimators, n_jobs=n_jobs, verbose=verbose, random_state=random_state)

    rf_model.fit(x_train, y_train)
    joblib.dump(rf_model, file_name+".joblib", compress=3)


def date_range(start_date, end_date):
	for n in range(int((end_date - start_date).days)):
		yield datetime.combine(start_date + timedelta(n), datetime.min.time())


def get_holidays_until(date):
	import requests
	import json
	holidays = []
	current_year = datetime.now().year
	year = datetime.strptime(date, '%d.%m.%Y').year
	if year > current_year:
		for i in range(current_year, year + 1):
			response = requests.get('https://date.nager.at/api/v2/PublicHolidays/' + str(i) + '/AT')
			holidays = [*holidays, *json.loads(response.text)]
	else:
		response = requests.get('https://date.nager.at/api/v2/PublicHolidays/' + str(year) + '/AT')
		holidays = json.loads(response.text)
	filtered_holidays = [str(datetime.combine(datetime.strptime(holiday['date'], '%Y-%m-%d').date(), datetime.min.time())) for holiday in holidays]
	return filtered_holidays


def random_forest_from_file(end_date_str='23.01.2024'):
	start_date = date.today()
	end_date = datetime.strptime(end_date_str, '%d.%m.%Y').date()
	dates_range = list(date_range(start_date, end_date))
	holidays = get_holidays_until(end_date_str)
	holidays = [1 if str(date) in holidays else 0 for date in dates_range]
	weekdays = [date.weekday() for date in dates_range]
	
	data = {
		'DayOfWeek': weekdays,
		'StateHoliday': holidays,
		'SchoolHoliday': holidays,
	}
    # data = {
    #     'Store': store,
    #     'DayOfWeek': dayofweek,
    #     'Customers': customers,
    #     'Open': open,
    #     'Promo': promo,
    #     'StateHoliday': stateholiday,
    #     'SchoolHoliday': schoolholiday,
    #     'StoreType': storetype,
    #     'Assortment': assortment,
    #     'CompetitionDistance': competitiondistance,
    #     'CompetitionOpenSinceMonth': competitionopensincemonth,
    #     'CompetitionOpenSinceYear': competitionopensinceyear,
    #     'Promo2': promo2,
    #     'Promo2SinceWeek': promo2sinceweek,
    #     'Promo2SinceYear': promo2sinceyear,
    #     'Year': year,
    #     'Month': month,
    #     'Day': day,
    #     'WeekOfYear': weekofyear,
    #     'CompetitionOpen': competitionopen,
    #     'PromoOpen': promoopen,
    #     'IsPromoMonth': ispromomonth
    # }

    # df = pd.DataFrame(data)
    # rf_model = joblib.load(file_name + ".joblib")
    # rf_predict = rf_model.predict(df)

    # return rf_predict

if __name__ == '__main__':
	random_forest_from_file()
