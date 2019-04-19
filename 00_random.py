#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 00_random.py 
@time: 2019-04-18 02:07
@description:
"""
import pandas as pd
import random
train_stock=pd.read_csv('input/TRAINSET_STOCK.csv')
df=pd.DataFrame()
tmp=[]
for code in train_stock.ts_code.unique():
    tmp.extend([code]*5)

df['ts_code']=tmp
df['trade_date']=[20190402,20190403,20190404,20190408,20190409]*len(train_stock.ts_code.unique())
pred=[]
for i in range(len(df)):
    pred.append(random.uniform(0,1))
df['p']=pred
# df['p']=df['p'].round(3)
df.to_csv('rn.csv',index=False)