#!/usr/bin/env python
# encoding: utf-8
"""
@author: 邹佳丽
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: zoujiali18@hotmail.com
@file: data_and_model.py
@time: 2020/9/14 17:58
@desc:
"""
'''
优化文本生成模型， 使用更大的模型
(1)使用双向LSTM 
(2) 使用one-hot编码，原文使用chat2int/voa_size 不合理
(2)training epochs from 20 to 50 
(3)decrease the batch size from 128 to 64 to give the network more of an opportunity to be updated and learn. 
'''
# Larger LSTM Network to Generate Text for Alice in Wonderland
import numpy
from keras.models import Sequential
from keras.layers import Dense, Lambda
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.layers import Bidirectional
import keras.backend as K
import tensorflow as tf

# todo: 根据需要（模型大小）占用 gpu 资源, 避免全部占用
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


def OneHot(input_dim=None, input_length=None):
    # Check if inputs were supplied correctly
    if input_dim is None:
        raise TypeError("input_dim or input_length is not set")
    # Helper method (not inlined for clarity)
    def _one_hot(x, num_classes):
        return K.one_hot(K.cast(x, 'uint8'), num_classes=num_classes)
    # Final layer representation as a Lambda layer
    return Lambda(_one_hot,
                  arguments={'num_classes': input_dim}, name='one-hot')


filename = "/home/lantone/zoujl/TXT_generate/LuXun.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
# X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = numpy.reshape(dataX, (n_patterns, seq_length))
# print(X.shape)
# print(X[0])

# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(OneHot(input_dim=n_vocab))  # TOdo:使用noe-hot编码
model.add(Bidirectional(LSTM(256))) # todo:使用双向 lstm
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.build(input_shape=[None, 100])
model.summary()
model.load_weights("/home/lantone/zoujl/TXT_generate/lx_model/"
                   "weights-improvement-38-2.8098-bigger.hdf5")  # todo: 载入第38轮的模型，接着训练

# define the checkpoint
filepath="/home/lantone/zoujl/TXT_generate/lx_model/" \
         "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)
# model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list, initial_epoch=38)todo: 模型接着断掉的第38轮接着训练

# Epoch 47/50
# 625791/625791 [==============================] - 1710s 3ms/step - loss: 2.5459
