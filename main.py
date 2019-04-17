#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: main.py 
@time: 2019-04-18 01:03
@description:
"""
import pandas as pd
train_stock=pd.read_csv('input/TRAINSET_STOCK.csv')
print(train_stock.info())
print(train_stock.name.value_counts(),len(train_stock.name.value_counts()))

dates=[20190402,20190402,20190403,20190404,20190405]