#pragma once
#include <Eigen/Dense>
#include <iostream>
#include "Module.h"

template <typename T = Eigen::MatrixXd>
class Linear :public Module<T> {
protected:
    const int in_features;
    const int out_features;
    const bool haveBias;
    Eigen::MatrixXd weight;
    Eigen::MatrixXd bias;

public:
    Linear(int in_features, int out_features, ActivationFunction* af = nullptr, LossFunction* lf = nullptr, bool haveBias = true);
    // 拷贝构造函数
    // TODO
    //Linear(const Linear& L);
    Eigen::MatrixXd forward(const T& x) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd pd, double learnRate = 1e-3) override;
    Eigen::MatrixXd getWeight() const;
    Eigen::MatrixXd getBias() const;

    void setWeight(const Eigen::MatrixXd& w);
    void setBias(const Eigen::MatrixXd& b);
};


template <typename T>
Linear<T>::Linear(const int input_features, const int output_features, ActivationFunction* af, LossFunction* lf, bool haveBias)
    :in_features(input_features), out_features(output_features), haveBias(haveBias), Module<T>(af, lf) {
    weight = Eigen::MatrixXd::Random(output_features, input_features);
    if (haveBias) {
        bias = Eigen::MatrixXd::Random(1, output_features);
    }
}

//Linear::Linear(const Linear& L) :in_features(L.in_features), out_features(L.out_features), haveBias(L.haveBias) {
//    weight = L.weight;
//    actfun = L.actfun;
//    lossfun = L.lf;
//    if (L.haveBias) {
//        bias = L.bias;
//    }
//}

template <typename T>
Eigen::MatrixXd Linear<T>::forward(const T& x) {
    int rows = (int)x.rows();
    int cols = (int)x.cols();

    if (cols != in_features)
        throw(std::exception(("当前输入矩阵的列数为" + std::to_string(cols) + "，它应当为" + std::to_string(in_features)).c_str()));
    Eigen::MatrixXd wt = weight.transpose();
    this->input = x;
    if (haveBias) {
        Eigen::MatrixXd Bias(rows, out_features);
        for (int i = 0; i < rows; i++) {
            Bias.row(i) = bias;
        }
        this->output = x * wt + Bias;
        //std::cout << "x: \n" << x << std::endl;
        //std::cout << "w: \n" << wt << std::endl;
        //std::cout << "b: \n" << Bias << std::endl;
        //std::cout << "y: \n" << this->output << std::endl;
    }
    else {
        this->output = x * wt;
    }
    if (this->actfun != nullptr)
        return this->actfun->forward(this->output);
    return this->output;
}

template <typename T>
Eigen::MatrixXd Linear<T>::getWeight() const {
    return weight;
}

template <typename T>
Eigen::MatrixXd Linear<T>::getBias() const {
    if (haveBias)
        return bias;
}

template <typename T>
Eigen::MatrixXd Linear<T>::backward(const Eigen::MatrixXd pd, double learnRate) {
    int inputNums = (int)this->input.rows();
    //int inputCols = (int)this->input.cols();
    //int output_rows = (int)this->output.rows();
    //int outputCols = (int)this->output.cols();
    //int pdRows = (int)pd.rows();
    //int pdCols = (int)pd.cols();

    Eigen::MatrixXd returnPd(inputNums, in_features);

    Eigen::MatrixXd changeWeight = Eigen::MatrixXd::Zero(out_features, in_features);

    if (this->actfun != nullptr) {
        Eigen::MatrixXd actfunPd = this->actfun->getPd(output);
        Eigen::MatrixXd temp(inputNums, out_features);
        for (int i = 0; i < inputNums; i++) {
            for (int j = 0; j < out_features; j++) {
                temp.row(i)[j] = pd.row(i)[j] * actfunPd.row(i)[j];
            }
        }
        returnPd = temp * weight;

        // Updata Wieght
        for (int i = 0; i < inputNums; i++) {
            Eigen::MatrixXd Diag = temp.row(i).asDiagonal();
            Eigen::MatrixXd X(out_features, in_features);
            for (int j = 0; j < out_features; j++) {
                X.row(j) = this->input.row(i);
            }
            changeWeight += Diag * X;
        }
        weight += -learnRate * changeWeight;

        // Updata Bias
        if (haveBias) {
            Eigen::MatrixXd changeBias = temp.colwise().sum();
            bias += -learnRate * changeBias;
        }
    }
    else {
        returnPd = pd * weight;
        // TODO 无激活函数
        // Updata Wieght
        for (int i = 0; i < inputNums; i++) {
            Eigen::MatrixXd Diag = pd.row(i).asDiagonal();
            Eigen::MatrixXd X(out_features, in_features);
            for (int j = 0; j < out_features; j++) {
                X.row(j) = this->input.row(i);
            }
            changeWeight += Diag * X;
        }
        weight += -learnRate * changeWeight;

        // Updata Bias
        if (haveBias) {
            Eigen::MatrixXd changeBias = pd.colwise().sum();
            bias += -learnRate * changeBias;
        }
    }
    return returnPd;
}

template <typename T>
void Linear<T>::setWeight(const Eigen::MatrixXd& w) {
    weight = w;
}

template <typename T>
void Linear<T>::setBias(const Eigen::MatrixXd& b) {
    bias = b;
}