# Neural Networks
A Simple Architecture for Neural Networks imitating pytorch  
这是一个通过C++实现的仿pytorch神经网络框架。

## 框架特点
1. 目前能实现任意规模的前馈神经网络BPNN；
2. 自动梯度计算由计算图实现，可自动计算大部分可导函数的梯度；
3. 目前已经实现了SGD+Momentum优化器；
4. 该框架不包含任何第三方库；
5. 支持自定义神经网络层；
6. ...

## 计划实现的功能
1. CNN卷积层；
2. CUDA运算加速
3. ...

## demo
目前main.cpp内包含一个由两层全连接层实现的鸢尾花分类的简单demo，其[训练集](http://download.tensorflow.org/data/iris_training.csv)和[测试集](http://download.tensorflow.org/data/iris_test.csv)已经包含在dataset目录下。

## Note
该框架使用了部分C++11特性，需要支持C++11才能运行
