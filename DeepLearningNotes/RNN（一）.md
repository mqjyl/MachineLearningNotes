---
title: RNN（一）
tags: RNN,LSTM,GRU
grammar_cjkRuby: true
---

#### 一、什么是RNN?

RNN（Recurrent Neural Network）是一类用于处理序列数据的神经网络。在CNN网络中的训练样本的数据为IID数据（独立同分布数据），所解决的问题也是分类问题或者回归问题或者是特征表达问题。但更多的数据是不满足IID的，如语言翻译，自动文本生成，它们是一个序列问题，包括时间序列和空间序列。