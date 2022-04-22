#include "LossFunction.h"

//double LossFunction::CrossEntropyLoss(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target) {
//    int inputSize[] = { input.rows(), input.cols() };
//    int targetSize[] = { target.rows(), target.cols() };
//
//    if (inputSize[0] != targetSize[0] || inputSize[1] != targetSize[1]) {
//        throw (std::exception(("target size is " + std::to_string(targetSize[0]) + "*" + std::to_string(targetSize[1]) +
//            ", but input size is " + std::to_string(inputSize[0]) + "*" + std::to_string(inputSize[1])).c_str()));
//    }
//    double s = 0;
//    for (int i = 0; i < inputSize[0]; i++) {
//        int j = 0;
//        while (j < inputSize[1] && target(i, j++) == 0);
//        s += std::log(input(i, --j));
//    }
//    return -(s / inputSize[0]);
//}
//
//double LossFunction::MESLoss(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target) {
//    int inputSize[] = { input.rows(), input.cols() };
//    int targetSize[] = { target.rows(), target.cols() };
//
//    if (inputSize[0] != targetSize[0] || inputSize[1] != targetSize[1]) {
//        throw (std::exception(("target size is " + std::to_string(targetSize[0]) + "*" + std::to_string(targetSize[1]) +
//            ", but input size is " + std::to_string(inputSize[0]) + "*" + std::to_string(inputSize[1])).c_str()));
//    }
//    double loss = 0;
//    for (int i = 0; i < inputSize[0]; i++) {
//        for (int j = 0; j < inputSize[1]; j++)
//            loss += pow(input(i, j) - target(i, j), 2);
//    }
//    return loss / (inputSize[0] * inputSize[1]);
//}
//
//double LossFunction::RMESLoss(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target) {
//    return pow(MESLoss(input, target), 0.5);
//}

double LossFunction::operator()(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target){
    return this->forward(input, target);
}

// MESLoss
Eigen::MatrixXd MESLoss::getPd(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target){
    double inputSize[] = { input.rows(), input.cols() };
    double targetSize[] = { target.rows(), target.cols() };

    if (inputSize[0] != targetSize[0] || inputSize[1] != targetSize[1]) {
        throw (std::exception(("目标矩阵大小为：" + std::to_string(targetSize[0]) + "*" + std::to_string(targetSize[1]) +
            "，但是输入矩阵大小为：" + std::to_string(inputSize[0]) + "*" + std::to_string(inputSize[1])).c_str()));
    }
    return 2 * (input - target);
}

double MESLoss::forward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target){
    double inputSize[] = { input.rows(), input.cols() };
    double targetSize[] = { target.rows(), target.cols() };
    
    if (inputSize[0] != targetSize[0] || inputSize[1] != targetSize[1]) {
        throw (std::exception(("目标矩阵大小为：" + std::to_string(targetSize[0]) + "*" + std::to_string(targetSize[1]) +
            "，但是输入矩阵大小为：" + std::to_string(inputSize[0]) + "*" + std::to_string(inputSize[1])).c_str()));
    }
    double loss = 0;
    for (int i = 0; i < inputSize[0]; i++) {
        for (int j = 0; j < inputSize[1]; j++)
            loss += pow(input(i, j) - target(i, j), 2);
    }
    return loss / (inputSize[0] * inputSize[1]);
}
