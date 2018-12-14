## Chinese Entity Recognize 2018-12

#### 1.preprocess



#### 2.explore

统计词汇、长度、实体的频率，条形图可视化，计算 slot_per_sent 指标

#### 3.represent



#### 4.build



#### 5.recognize

predict() 填充为定长序列、每句返回 (word, pred) 的二元组

#### 6.interface

merge() 将 BIO 标签组合为实体，response() 返回 json 字符串