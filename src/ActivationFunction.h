#pragma once
#include <Eigen/Dense>
#include <cmath>


class ActivationFunction {
protected:
    Eigen::MatrixXd output;
    Eigen::MatrixXd pd;
public:
    // 计算偏导数（Partial derivative, pd）
    virtual Eigen::MatrixXd getPd(const Eigen::MatrixXd& x) = 0;
    // 计算激活函数后的值
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& x) = 0;
    Eigen::MatrixXd operator()(const Eigen::MatrixXd& x);

    //Eigen::MatrixXd ReLU(const Eigen::MatrixXd& x);
    //Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x);
    //Eigen::MatrixXd softmax(const Eigen::MatrixXd& x);
    //Eigen::MatrixXd tanh(const Eigen::MatrixXd& x);
};

class sigmoid: public ActivationFunction {
    Eigen::MatrixXd getPd(const Eigen::MatrixXd& x) override;
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& x) override;
};

//namespace ActivationFunction {
//    Eigen::MatrixXd ReLU(const Eigen::MatrixXd& x);
//    Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x);
//    Eigen::MatrixXd softmax(const Eigen::MatrixXd& x);
//    Eigen::MatrixXd tanh(const Eigen::MatrixXd& x);
//}

