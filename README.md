## Chinese Entity Recognize 2018-12

#### 1.preprocess

generate() 根据 template 采样实体进行填充、生成数据，() 表示可省去

label_sent() 对 extra 查找 entity 索引进行标注、添加到训练数据

#### 2.explore

统计词汇、长度、实体的频率，条形图可视化，计算 slot_per_sent 指标

#### 3.represent

label2ind() 增设标签 N，add_buf() 再对 cnn_sent 头部、尾部进行填充

#### 4.build

train 80% / dev 20% 划分，通过 cnn、rnn 构建实体识别模型

#### 5.recognize

predict() 填充为定长序列、每句返回 (word, pred) 的二元组

#### 6.verify

merge() 将 BIO 标签组合为实体，response() 返回 json 字符串