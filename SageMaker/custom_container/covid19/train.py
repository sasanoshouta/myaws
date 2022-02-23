#!/usr/bin/env python3
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import metrics

DATA_PATH = "../ml/input/data/yamazo-qiita-channel/made_data_08_31.csv" # S3からDownloadされる学習データ
MODEL_PATH = "../ml/model/model_1.h5"                        # S3へUploadされるモデル

NAME_JP_NUM = 26
use_data = pd.read_csv(DATA_PATH, index_col=0)

pref_name_df = use_data[['name_jp', 'labeled_name_jp']].drop_duplicates()
alart_name_df = use_data[['alart', 'labeled_alart']].drop_duplicates().reset_index(drop=True)
# day_name_df = use_data[['day', 'labeled_day']].drop_duplicates().reset_index(drop=True)
X = use_data.drop(['name_jp', 'alart', 'patients', 'count_vacctine'], axis=1)

X_train = X[X['date'] <= '2021/07/15'].reset_index(drop=True)
X_test = X[X['date'] >= '2021/08/01'].reset_index(drop=True)
X_val = X[(X['date'] > '2021/07/15') & (X['date'] < '2021/08/01')].reset_index(drop=True)

tokyo_X_train = X_train[X_train['labeled_name_jp'] == NAME_JP_NUM].reset_index(drop=True)
tokyo_X_test = X_test[X_test['labeled_name_jp'] == NAME_JP_NUM].reset_index(drop=True)
tokyo_X_val = X_val[X_val['labeled_name_jp'] == NAME_JP_NUM].reset_index(drop=True)
tokyo_y_train = tokyo_X_train[['day_patient']].reset_index(drop=True)
tokyo_y_test = tokyo_X_test[['day_patient']].reset_index(drop=True)
tokyo_y_val = tokyo_X_val[['day_patient']].reset_index(drop=True)

toplotx = tokyo_X_test['date']

tokyo_X_train.drop(['date', 'labeled_name_jp', 'day_patient'], axis=1, inplace=True)
tokyo_X_test.drop(['date', 'labeled_name_jp', 'day_patient'], axis=1, inplace=True)
tokyo_X_val.drop(['date', 'labeled_name_jp', 'day_patient'], axis=1, inplace=True)

tokyo_X_train = np.array(tokyo_X_train).reshape(tokyo_X_train.shape[0], tokyo_X_train.shape[1], 1)
tokyo_X_test = np.array(tokyo_X_test).reshape(tokyo_X_test.shape[0], tokyo_X_test.shape[1], 1)
tokyo_X_val = np.array(tokyo_X_val).reshape(tokyo_X_val.shape[0], tokyo_X_val.shape[1], 1)
tokyo_y_train = np.array(tokyo_y_train).reshape(tokyo_y_train.shape[0], tokyo_y_train.shape[1], 1)
tokyo_y_test = np.array(tokyo_y_test).reshape(tokyo_y_test.shape[0], tokyo_y_test.shape[1], 1)
tokyo_y_val = np.array(tokyo_y_val).reshape(tokyo_y_val.shape[0], tokyo_y_val.shape[1], 1)

def train(tokyo_X_train, tokyo_y_train, tokyo_X_val, tokyo_y_val):
    loss = 'mean_squared_error'
    val_loss = 'val_mean_squared_error'
    model = Sequential()
    model.add(LSTM(128, activation = 'relu', input_shape = (tokyo_X_train.shape[1], 1)))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss= loss, optimizer = Adam(), metrics=[metrics.mse])
    
    loss_history = list()
    val_loss_history = list()
    EPOCHS = 50
    epochs = 100
    
    callback = EarlyStopping(monitor='loss', patience=100)

    print('TRAIN START')
    for i in range(EPOCHS):
        history = model.fit(tokyo_X_train, tokyo_y_train, epochs=epochs, verbose=0, validation_data=(tokyo_X_val, tokyo_y_val), callbacks=[callback], batch_size=14)
        history_size = len(history.history['loss'])
        diff_loss = abs(history.history['val_loss'][history_size - 1] - history.history['loss'][history_size - 1])
        if ((i + 1) % 5 == 0) or (i + 1 == 1): 
            print('{} / {} : loss: {:.4f} | val_loss: {:.4f} | MAE: {:.4f} | val_MAE: {:.4f} | diff_loss: {:.4f}'.format(i + 1, EPOCHS, history.history['loss'][history_size - 1], history.history['val_loss'][history_size - 1],
            history.history[loss][history_size - 1], history.history[val_loss][history_size - 1], diff_loss))
        elif history_size != epochs:
            print('{} / {} : loss: {:.4f} | val_loss: {:.4f} | MAE: {:.4f} | val_MAE: {:.4f} | diff_loss: {:.4f}'.format(i + 1, EPOCHS, history.history['loss'][history_size - 1], history.history['val_loss'][history_size - 1],
            history.history[loss][history_size - 1], history.history[val_loss][history_size - 1], diff_loss))
            print('Early stopping is worked')
        else:
            pass
        
        for j in range(history_size):
            loss_history.append(history.history['loss'][j])
            val_loss_history.append(history.history['val_loss'][j])
            
    print('TRAIN FINISHED')
    
    model.save(MODEL_PATH)

train(tokyo_X_train, tokyo_y_train, tokyo_X_val, tokyo_y_val)