#!/usr/bin/env python
# encoding: utf-8
"""
@author: 邹佳丽
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: zoujl@zjlantone.com
@company: LANTONE
@file: songCi_data_preprocess.py
@time: 2020/9/10 16:59
@desc:
"""
'''
由于英文生成效果方便看，利用已经调试好的LSTM代码，在中文文本上尝试 
数据：https://github.com/chinese-poetry/chinese-poetry/tree/master/ci
说明：全宋词
'''
import json
import os
# todo: 将json数据格式 整理成 txt
# {"author": "王禹",
#     "paragraphs": [
#       "雨恨云愁，江南依旧称佳丽。",
#       "水村渔市。"],
#     "rhythmic": "点绛唇"}

# 点绛唇 王禹
# 雨恨云愁，江南依旧称佳丽
# 水村渔市
root = "C:\\Users\\zouji\\Desktop\\TXT_generate\\ci\\"
out = "C:\\Users\\zouji\\Desktop\\TXT_generate\\ci\\ci_out\\"
ci_txt = "C:\\Users\\zouji\\Desktop\\TXT_generate\\ci.txt"

def format_json(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as fr:
        with open(out_path, "w", encoding="utf-8") as fw:
            data = json.load(fr)
            for i in data:
                author = i["author"]
                paragraphs = i["paragraphs"]
                paragraphs = "\n".join([i for i in paragraphs])
                # print(paragraphs)
                rhythmic = i["rhythmic"]
                content = author + "\t" + rhythmic + "\n" + paragraphs + "\n\n"
                print(content)
                fw.write(content)


# for file in os.listdir(root):
#     if file.startswith("ci") and file.endswith("json"):
#         in_path = root + file
#         out_path = out + file.replace("json", "txt")
#         format_json(in_path, out_path)

with open(ci_txt, "w", encoding="utf-8") as fw:
    for file in os.listdir(out):
        with open(out+file, "r", encoding="utf-8") as fr:
            data = fr.read()
            fw.write(data)