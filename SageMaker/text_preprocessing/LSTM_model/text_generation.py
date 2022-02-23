import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn import metrics as skmet

# 単語のベクトル化処理
train = []
for i in range(len(text_to_lstm)):
    for j in range(len(text_to_lstm[i])):
        train.append(text_to_lstm[i][j])
        
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train)
char_indices = tokenizer.word_index
indices_char = dict([(value, key) for (key, value) in char_indices.items()])
pre_vec = tokenizer.texts_to_sequences(train)

texts_vec = []
thresh_words = 3
for line in range(len(pre_vec)):
    if len(pre_vec[line]) > thresh_words:
        texts_vec.append(pre_vec[line])

seq_length = 1
x = []
y = []
for line in range(len(texts_vec)):
    for i in range(len(texts_vec[line])-seq_length):
        x.append(texts_vec[line][i:i+seq_length])
        y.append(texts_vec[line][i+seq_length])
        
x = np.reshape(x,(len(x),seq_length,1))
y = to_categorical(y, len(char_indices)+1)


# モデル構築
loss = 'categorical_crossentropy'
val_loss = 'val_mean_squared_error'
model = Sequential()
model.add(LSTM(128, activation = 'relu', input_shape = (seq_length, 1)))
model.add(Dropout(0.5))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss= loss, optimizer = Adam(), metrics=['accuracy'])

# 学習
loss_history = list()
val_loss_history = list()
EPOCHS = 10
epochs = 100
# steps_per_epoch = 140

callback = EarlyStopping(monitor='loss', patience=100)

print('TRAIN START')
for i in range(EPOCHS):
    history = model.fit(x, y, epochs=epochs, verbose=0, callbacks=[callback])
    history_size = len(history.history['loss'])
    if ((i + 1) % 5 == 0) or (i + 1 == 1): 
        print('{} / {} : loss: {:.4f}  | accuracy: {:.4f}'.format(i + 1, EPOCHS, history.history['loss'][history_size - 1],history.history['accuracy'][history_size - 1]))
    
    elif history_size != epochs:
        print('{} / {} : loss: {:.4f}  | accuracy: {:.4f}'.format(i + 1, EPOCHS, history.history['loss'][history_size - 1],history.history['accuracy'][history_size - 1]))
        print('Early stopping is worked')
        
    else:
        pass
    
    for j in range(history_size):
        loss_history.append(history.history['loss'][j])
        val_loss_history.append(history.history['accuracy'][j])
    
model.save('model.h5')
print('TRAIN FINISHED')

new_model = models.load_model('pdf.h5')

# 推論
# 作成する試験項目の長さ
test_sentence_length = 30
# 作成する試験項目数
test_case_num = 10
# 作成した試験項目の保存先　→　key:先頭の単語　value:試験項目
created_test_case = dict()
# 先頭の単語を指定する乱数（数が多い為）
use_word_num = np.random.randint(0, len(important_vocab_list), test_case_num)

for num in range(test_case_num):
    test_case = []
    first_word = important_vocab_list[use_word_num[num]]
    test_case.append(first_word)
    first_word = char_indices[first_word]
    first_word = np.reshape(first_word, (1, seq_length, 1))
    first_word = first_word / float(len(char_indices))
    tmp = model.predict(first_word, verbose=0)
    index = np.argmax(tmp)
    result = indices_char[index]
    test_case.append(result)
    
    for i in range(test_sentence_length):
        index = np.reshape(index, (1, seq_length, 1))
        index = index / float(len(char_indices))
        tmp = model.predict(index, verbose=0)
        index = np.argmax(tmp)
        result = indices_char[index]
        test_case.append(result)
    created_test_case[test_case[0]] = " ".join(test_case)
