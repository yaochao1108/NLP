
# coding: utf-8



import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow.keras.layers as layers




def check(c):
    return '\u4e00' <= c <= '\u9fa5'




def get_words(sent_ids):
    return ' '.join([id2word.get(i, '?') for i in sent_ids])




neg = pd.read_csv("data/neg.csv",header = None)




pos = pd.read_csv("data/pos.csv",error_bad_lines=False,header = None)




li = len(neg)*[0]+len(pos)*[1]




random.shuffle(li)




word2id = {}
n = 1
for words in neg[0]:
    for word in words:
        if check(word):
            if word not in word2id:
                word2id[word] = n
                n+=1
for words in pos[0]:
    for word in words:
        if check(word):
            if word not in word2id:
                word2id[word] = n
                n+=1




index = list(range(1,len(word2id)+1))




random.shuffle(index)




n = 0
for word in word2id:
    word2id[word] = index[n]
    n+=1




id2word = {v:k for k, v in word2id.items()}




words_list = []
neg_num = 0
pos_num = 0
for i in li:
    if i == 0:
        x = [word for word in neg[0][neg_num]]
        y = list(filter(check,x))
        z = list(map(lambda x:word2id[x],y))
        words_list.append(z)
        neg_num+=1
    if i == 1:
        x = [word for word in pos[0][pos_num]]
        y = list(filter(check,x))
        z = list(map(lambda x:word2id[x],y))
        words_list.append(z)
        pos_num+=1




words_list = keras.preprocessing.sequence.pad_sequences(
    words_list, value=0,
    padding='post', maxlen=128
)




train_x = words_list[:int(len(words_list)*0.7)]
train_y = li[:int(len(words_list)*0.7)]

test_x = words_list[int(len(words_list)*0.7):]
test_y = li[int(len(words_list)*0.7):]




vocab_size = len(word2id)+1
model = keras.Sequential()
model.add(layers.Embedding(vocab_size, 16))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(16, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])




history = model.fit(train_x,train_y,
                   epochs=40, batch_size=1024,
                   validation_data=(test_x, test_y),
                   verbose=1)
