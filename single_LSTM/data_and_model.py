#!/usr/bin/env python
# encoding: utf-8
"""
@author: 邹佳丽
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: zoujiali18@hotmail.com
@file: data_and_model.py
@time: 2020/9/10 11:28
@desc:
"""
'''
刚学完 Lstm 原理, 用LSTM做文本生成
文本数据：爱丽丝梦游仙境
数据下载： http://www.gutenberg.org/cache/epub/11/pg11.txt

由于英文生成效果方便看，利用已经调试好的LSTM代码，在中文文本上尝试 
数据：https://github.com/chinese-poetry/chinese-poetry/tree/master/ci
说明：全宋词
'''
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# todo: load ascii text and covert to lowercase
# filename = "wonderland.txt"
# raw_text = raw_text.lower()
# raw_text = "".join([s for s in raw_text.splitlines(True) if s.strip()])
# print(raw_text)
filename = "LuXun.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
# print(raw_text)
# todo: 文字变数字，create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
print(char_to_int)
# sumarize TXT
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
# TODO：生成X,Y  prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        # in ： start of this project gutenberge book alice's adventures in wonderland *** alice's adventures in won
        # out： d
        # in ： tart of this project gutenberg ebook alice's adventures in wonderland *** alice's adventures in wond
        # out： e
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# todo:数据格式满足模型(LSTM需要的格式)
# reshape X to be [samples, time steps, features] expected by an LSTM network.
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# Next we need to rescale the integers to the range 0-to-1 to make the patterns easier to learn by the LSTM network
# that uses the sigmoid activation function by default.
# normalize
X = X / float(n_vocab)
# one hot encode the output variable 将整型的类别标签转为onehot编码
y = np_utils.to_categorical(dataY)
# print(y)


# TODO：定义LSTM 模型
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# 打印模型结构及参数情况
print(model.summary())
# define the checkpoint
# todo: 保存模型
filepath = "./lx_model/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# todo:模型训练
# model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
model.fit(X, y, epochs=50, batch_size=128, callbacks=callbacks_list)