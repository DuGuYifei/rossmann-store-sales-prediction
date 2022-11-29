import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from keras.layers import Dense, InputLayer
from keras.layers import SimpleRNN, LSTM
from keras.models import Sequential
import keras.backend as K

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
submission = pd.read_csv('dataset/sample_submission.csv')
store = pd.read_csv('dataset/store.csv')

train['Date'] = pd.to_datetime(train['Date'])
train['Year'] = train['Date'].dt.year
train['Month'] = train['Date'].dt.month
train['Day'] = train['Date'].dt.day

train.StateHoliday.replace({'0': 0,
                            'a': 1,
                            'b': 2,
                            'c': 3}, inplace=True)

test['Date'] = pd.to_datetime(test['Date'])
test['Year'] = test['Date'].dt.year
test['Month'] = test['Date'].dt.month
test['Day'] = test['Date'].dt.day
test['StateHoliday'].unique()

test.StateHoliday.replace({'0': 0,
                           'a': 1}, inplace=True)

test.Open.fillna(1, inplace=True)

Labeling = LabelEncoder()
store['StoreType'] = Labeling.fit_transform(store['StoreType'])
store['Assortment'] = Labeling.fit_transform(store['Assortment'])


store[['FirstQuarter', 'SecondQuarter', 'ThirdQuarter', 'FourtQquarter']
      ] = store['PromoInterval'].str.split(',', expand=True)
store.drop('PromoInterval', axis=1, inplace=True)

store.FirstQuarter.replace({'Jan': 1,
                            'Feb': 2,
                            'Mar': 3}, inplace=True)

store.SecondQuarter.replace({'Apr': 1,
                            'May': 2,
                             'Jun': 3}, inplace=True)

store.ThirdQuarter.replace({'Jul': 1,
                            'Aug': 2,
                            'Sept': 3}, inplace=True)

store.FourtQquarter.replace({'Oct': 1,
                            'Nov': 2,
                             'Dec': 3}, inplace=True)

store.fillna(0, inplace=True)

train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')

train.sort_values(by=['Store', 'Date'], inplace=True)
test.sort_values(by=['Store', 'Date'], inplace=True)

train.set_index(['Date'], inplace=True)
test.set_index(['Date'], inplace=True)

scaler = MinMaxScaler()

train_x = train.drop(['Sales', 'Customers'], axis=1)
train_y = train[['Sales']]
#train_x_col = train_x.columns
#train_y_col = train_y.columns

train_x[train_x.columns] = scaler.fit_transform(train_x[train_x.columns])
test[train_x.columns] = scaler.transform(test[train_x.columns])

train_x = train_x.values
train_y = train_y.values
test = test.values

test = test.reshape(test.shape[0], test.shape[1], 1)


x_train, x_val, y_train, y_val = train_test_split(
    train_x, train_y, test_size=.2)


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

my_batch = 64
my_epoch = 200
my_past = 12
my_split = 0.5
my_neuron = 500  # RNN, LSTM parameter
my_shape = (my_past, 1)

K.clear_session()
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(train_x.shape[1], 1)))
# model.add(SimpleRNN(my_neuron))
model.add(tf.compat.v1.keras.layers.CuDNNLSTM(my_neuron, go_backwards=True))
model.add(tf.keras.layers.Dense(my_neuron, activation='relu'))
model.add(tf.keras.layers.Dropout(.3))
model.add(tf.keras.layers.Dense(my_neuron, activation='relu'))
model.add(tf.keras.layers.Dropout(.3))
model.add(tf.keras.layers.Dense(1, activation='linear'))

model.compile(optimizer='rmsprop', loss='mse', metrics=['acc'])
model.fit(x_train, y_train, batch_size=my_batch, validation_split=.2,
          epochs=my_epoch, use_multiprocessing=True, verbose=0)

# RNN Evaluate
score = model.evaluate(x_val, y_val, verbose=1)
print('Loss:' + format(score[0], "1.3f"))

pred = model.predict(test)

submission['Sales'] = pred

submission.to_csv('./sample.csv', index=False)
