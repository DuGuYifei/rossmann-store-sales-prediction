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
