# Neural Networks
A Simple Architecture for Neural Networks imitating pytorch  
这是一个通过C++实现的仿pytorch神经网络框架，其中的矩阵运算通过[Eigen](https://eigen.tuxfamily.org/)实现。

## 框架特点
1. 能实现任意规模的前馈神经网络BPNN；
2. 自动梯度计算由计算图实现，可自动计算可导函数的梯度；
3. 目前已经实现了SGD+Momentum优化器；
4. ...

## 计划实现的功能
1. CNN卷积层；
2. Tensor类还只支持二维数据；
3. ...

## demo
目前main.cpp内包含一个由两层全连接层实现的鸢尾花分类的简单demo，其[训练集](http://download.tensorflow.org/data/iris_training.csv)和[测试集](http://download.tensorflow.org/data/iris_test.csv)已经包含在dataset目录下。

## Note
运行此文件前需要下载[Eigen](https://eigen.tuxfamily.org/)并通过CMake配置
