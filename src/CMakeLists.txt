# CMakeList.txt: BPNNProject 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

# 将源代码添加到此项目的可执行文件。

aux_source_directory(. SRC_LIST)
add_executable(main ${SRC_LIST})

# add_executable(main main.cpp ActivationFunction.cpp Adam.cpp Linear.cpp LossFunction.cpp Module.cpp NeuralNetworks.cpp Optimizer.cpp)