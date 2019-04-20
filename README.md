# Ashare-Industry-Trends
公开新闻预测A股行业板块动向 

## 任务描述
要求开发者利用历史公开新闻数据，完成目标自然语言建模任务

- 公开新闻指和鲸提供的整理好的2014年至今历史新闻联播文本
- 预测目标：中国A股34个申万行业板块
- 预测内容：预测行业板块未来5个交易日收盘价涨的概率 
- 利用2014年-2019年4月1日期间的历史数据，预测4月2、3、4、8、9日的A股行业板块上涨的概率

## 数据
- ts_code：行业板块代码
- trade_date：交易日期
- name：行业板块名称
- open：交易日的开盘价
- low：交易日的最低价
- high：交易日的最高价
- close：交易日的收盘价
- change：涨跌额
- pct_change：涨跌幅
- vol：成交量
- amount：成绩额
- pe：市盈率
- pb：市净率
- y：是否涨 1表示涨，0表示跌

## 特征工程
- 威廉指标（Williams %R）或简称W%R，是一个振荡指标，是依股价的摆动点来度量股票／指数是否处于超买或超卖的现象。它衡量多空双方创出的峰值（最高价）距每天收市价的距离与一定时间内（如7天、14天、28天等）的股价波动范围的比例，以提供出股市趋势反转的讯号。
> 威廉指数：（最高价-收盘价）/(最高价-最低价)*100
- 相对强弱指数（RSI）是通过比较一段时期内的平均收盘涨数和平均收盘跌数来分析市场买沽盘的意向和实力，从而作出未来市场的走势。
> RSI＝[上升平均数÷(上升平均数＋下跌平均数)]×100
## 参考文献
- [Stock Trend Prediction with Technical Indicators using SVM](http://cs229.stanford.edu/proj2014/Xinjie%20Di,%20Stock%20Trend%20Prediction%20with%20Technical%20Indicators%20using%20SVM.pdf)
- [相对强弱指标](https://wiki.mbalib.com/wiki/%E7%9B%B8%E5%AF%B9%E5%BC%BA%E5%BC%B1%E6%8C%87%E6%A0%87)