#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: en.py 
@time: 2019-05-01 22:27
@description:
"""
import pandas as pd
import numpy as np

lgb = pd.read_csv('output/lgb_0.9145825686356523.csv')
xgb = pd.read_csv('output/xgb_0.8515866227793623.csv')

lgb['p'] = lgb['p'] * 0.3 + xgb['p'] * 0.7
# lgb['p'] = np.round(lgb['p'],2)


lgb.to_csv('output/en37.csv', index=False)
