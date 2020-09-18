#!/usr/bin/env python
# encoding: utf-8
"""
@author: 邹佳丽
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: zoujiali18@hotmail.com
@file: generate_text.py
@time: 2020/9/10 14:42
@desc:
"""
'''
Generating Text with an LSTM Network
'''
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import sys

# todo: load ascii text and covert to lowercase
filename = "LuXun.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
# raw_text = raw_text.lower()
# # todo-zjl:删除文中换行
# raw_text = "".join([s for s in raw_text.splitlines(True) if s.strip()])
# print(raw_text)
# todo: 文字变数字，create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# print(char_to_int)
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
# TODO：定义LSTM 模型
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

print(model.summary())

# load the network weights
# filename = "./model/weights-improvement-20-1.9105.hdf5" # 爱丽丝梦游仙境 20轮
# filename = "./ci_model/weights-improvement-100-4.1233.hdf5" # 宋词 100轮
filename = "./lx_model/weights-improvement-50-5.0177.hdf5" # 鲁迅全集  50轮

model.load_weights(filename)
# 整数到字母的映射
int_to_char = dict((i, c) for i, c in enumerate(chars))

# TODO：文本生成
# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("输入：")
print(''.join([int_to_char[value] for value in pattern]), "\"")
print("输出：")
# generate characters
for i in range(200):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction) # todo: 修改 解码方法
    result = int_to_char[index]
    # 打印到一行上
    # sys.stdout.write(result)
    print(result, end='')
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\n生成完毕！")