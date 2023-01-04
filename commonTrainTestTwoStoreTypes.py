from common import *
from sklearn.model_selection import train_test_split

train = pd.read_csv('dataset/train.csv', low_memory=False)
store = pd.read_csv('dataset/store2.csv', low_memory=False)

store.fillna(0, inplace=True)

train = train.loc[train.Open != 0]
train = train.loc[train.Sales > 0].reset_index(drop=True)
train.drop(['Customers'], axis=1, inplace=True)

train = pd.merge(train, store, on='Store')

#'StoreType' unique ['c' 'a']
#'Assortment' unique ['a' 'c' 'b']

###data_<StoreType>_<Assortment>

data = train.groupby(['StoreType'])
data_a = data.get_group('a')
data_b = data.get_group('b')

data_a = data_a.groupby(['Assortment'])
#unique ['a', 'c']
data_a_a = data_a.get_group('a')
data_a_c = data_a.get_group('c')

data_b= data_b.groupby(['Assortment'])
#unique ['a', 'b', 'c']
data_b_a = data_b.get_group('a')
data_b_b = data_b.get_group('b')
data_b_c = data_b.get_group('c')


###Train & test

#data_a_a
x_data_a_a_train, x_data_a_a_test = train_test_split(data_a_a, test_size=0.02, random_state=10)
x_data_a_a_train.reset_index(drop=True, inplace=True)
x_data_a_a_test.reset_index(drop=True, inplace=True)
features_create(x_data_a_a_train)
features_create(x_data_a_a_test)
y_data_a_a_train = x_data_a_a_train.pop('Sales')
y_data_a_a_test = x_data_a_a_test.pop('Sales')

#data_a_c
x_data_a_c_train, x_data_a_c_test = train_test_split(data_a_c, test_size=0.02, random_state=10)
x_data_a_c_train.reset_index(drop=True, inplace=True)
x_data_a_c_test.reset_index(drop=True, inplace=True)
features_create(x_data_a_c_train)
features_create(x_data_a_c_test)
y_data_a_c_train = x_data_a_c_train.pop('Sales')
y_data_a_c_test = x_data_a_c_test.pop('Sales')

#data_b_a
x_data_b_a_train, x_data_b_a_test = train_test_split(data_b_a, test_size=0.02, random_state=10)
x_data_b_a_train.reset_index(drop=True, inplace=True)
x_data_b_a_test.reset_index(drop=True, inplace=True)
features_create(x_data_b_a_train)
features_create(x_data_b_a_test)
y_data_c_a_train = x_data_b_a_train.pop('Sales')
y_data_c_a_test = x_data_b_a_test.pop('Sales')

#data_b_b
x_data_b_b_train, x_data_b_b_test = train_test_split(data_b_b, test_size=0.02, random_state=10)
x_data_b_b_train.reset_index(drop=True, inplace=True)
x_data_b_b_test.reset_index(drop=True, inplace=True)
features_create(x_data_b_b_train)
features_create(x_data_b_b_test)
y_data_c_a_train = x_data_b_b_train.pop('Sales')
y_data_c_a_test = x_data_b_b_test.pop('Sales')

#data_c_c
x_data_b_c_train, x_data_b_c_test = train_test_split(data_b_c, test_size=0.02, random_state=10)
x_data_b_c_train.reset_index(drop=True, inplace=True)
x_data_b_c_test.reset_index(drop=True, inplace=True)
features_create(x_data_b_c_train)
features_create(x_data_b_c_test)
y_data_c_c_train = x_data_b_c_train.pop('Sales')
y_data_c_c_test = x_data_b_c_test.pop('Sales')