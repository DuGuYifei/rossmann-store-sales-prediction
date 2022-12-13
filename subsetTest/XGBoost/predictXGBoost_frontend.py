from datetime import date, datetime, timedelta
import pandas as pd
from predictXGboost import predict

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


def predict_XGBoost_assumption(store_id, end_date_str='23.01.2024', save_file=False, customer_num=7000, with_customer=False):
	start_date = date.today()
	end_date = datetime.strptime(end_date_str, '%d.%m.%Y').date()
	dates_range = list(date_range(start_date, end_date))
	holidays = get_holidays_until(end_date_str)
	holidays = [1 if str(date) in holidays else 0 for date in dates_range]
	weekdays = [date.weekday() for date in dates_range]
	dates_list = [f"{d.year}-{d.month}-{d.day}" for d in dates_range]
	promo = [1 if d.month == 1 or d.month == 7 else 0 for d in dates_range]
	open = [1 for _ in dates_range]
	customers = [customer_num for _ in dates_range]

	data = {
		'Date': dates_list,
		'DayOfWeek': weekdays,
		'Customers': customers,
		'Open': open,
		'Promo': promo,
		'StateHoliday': holidays,
		'SchoolHoliday': holidays
	}

	store = pd.read_csv('../../dataset/store.csv', low_memory=False)
	store.fillna(0, inplace=True)
	store_info = store.loc[store['Store'] == store_id]
	store_info.reset_index(drop=True, inplace=True)

	data = pd.DataFrame(data)
	data.insert(loc=0, column="PromoInterval", value=store_info['PromoInterval'])
	data.insert(loc=0, column="Promo2SinceYear", value=int(store_info['Promo2SinceYear']))
	data.insert(loc=0, column="Promo2SinceWeek", value=int(store_info['Promo2SinceWeek']))
	data.insert(loc=0, column="Promo2", value=int(store_info['Promo2']))
	data.insert(loc=0, column="CompetitionOpenSinceYear", value=int(store_info['CompetitionOpenSinceYear']))
	data.insert(loc=0, column="CompetitionOpenSinceMonth", value=int(store_info['CompetitionOpenSinceMonth']))
	data.insert(loc=0, column="CompetitionDistance", value=int(store_info['CompetitionDistance']))
	data.insert(loc=0, column="Assortment", value=store_info['Assortment'])
	data.insert(loc=0, column="StoreType", value=store_info['StoreType'])
	data.insert(loc=0, column="Store", value=int(store_info['Store']))

	data['StoreType'].fillna(data['StoreType'][0], inplace=True)
	data['Assortment'].fillna(data['Assortment'][0], inplace=True)
	data['PromoInterval'].fillna(data['PromoInterval'][0], inplace=True)

	if not with_customer:
		return predict(data, data['StoreType'][0], data['Assortment'][0], save_file)
	# below are in the other brach ELT-42 xgboost with customer, but I think we will not use it,
	# because:
	#  1. user doesn't know customer number when he want to predict
	#  2. error rate without customer is good enough
	# else:
	#	return predict(data, data['StoreType'][0], data['Assortment'][0])


predict_XGBoost_assumption(1, '25.02.2023')