# -*- coding: utf-8 -*-
"""
@author: tz_zs
@file: nlp_data_demo.py
@time: 2018/7/11 10:42
"""
from tutorials.rnn.ptb import reader

train_data, valid_data, test_data, _ = reader.ptb_raw_data("./source_data/rnn_tz/data/")
print(len(train_data))
print(train_data[:100])

reader.ptb_producer(raw_data=train_data, batch_size=4, num_steps=5)

