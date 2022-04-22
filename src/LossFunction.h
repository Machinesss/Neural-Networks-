#pragma once
#include <Eigen/Dense>
#include <cmath>
#include "ActivationFunction.h"

class LossFunction {
public:
    // ����ƫ������Partial derivative, pd��
    virtual Eigen::MatrixXd getPd(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target) = 0;
    // ������ʧֵ
    virtual double forward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target) = 0;
    double operator() (const Eigen::MatrixXd& input, const Eigen::MatrixXd& target);
};

class MESLoss : public LossFunction {
public:
    Eigen::MatrixXd getPd(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target) override;
    double forward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target) override;
};

//namespace LossFunction {
//    double CrossEntropyLoss(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target);
//    double MESLoss(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target);
//    double RMESLoss(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target);
//}