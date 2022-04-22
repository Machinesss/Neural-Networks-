#include "ActivationFunction.h"

// ActivationFunction
Eigen::MatrixXd ActivationFunction::operator()(const Eigen::MatrixXd& x) {
    return forward(x);
}

// sigmoid
Eigen::MatrixXd sigmoid::getPd(const Eigen::MatrixXd& x){
    int rows = (int)output.rows();
    int cols = (int)output.cols();
    Eigen::MatrixXd One = Eigen::MatrixXd::Ones(cols, 1);
    pd = Eigen::MatrixXd::Zero(rows, cols);
    // TODO ? 
    for (int i = 0; i < rows; i++) {
        Eigen::MatrixXd output_i = output.row(i).transpose();
        pd.row(i) = (output_i.asDiagonal() * (One - output_i)).transpose();
    }
    return pd;
}

Eigen::MatrixXd sigmoid::forward(const Eigen::MatrixXd& x){
    int rows = (int)x.rows();
    int cols = (int)x.cols();
    Eigen::MatrixXd y = x;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            y(i, j) = 1 / (1 + exp(-y(i, j)));
        }
    }
    output = y;
    return output;
}

//Eigen::MatrixXd ActivationFunction::ReLU(const Eigen::MatrixXd& x) {
//    int rows = (int)x.rows();
//    int cols = (int)x.cols();
//    Eigen::MatrixXd y = x;
//    for (int i = 0; i < rows; i++) {
//        for (int j = 0; j < cols; j++) {
//            y(i, j) = y(i, j) < 0 ? 0 : y(i, j);
//        }
//    }
//    return y;
//}

//Eigen::MatrixXd ActivationFunction::sigmoid(const Eigen::MatrixXd& x) {
//    int rows = (int)x.rows();
//    int cols = (int)x.cols();
//    Eigen::MatrixXd y = x;
//    for (int i = 0; i < rows; i++) {
//        for (int j = 0; j < cols; j++) {
//            y(i, j) = 1 / (1 + exp(-y(i, j)));
//        }
//    }
//    return y;
//}

//Eigen::MatrixXd ActivationFunction::softmax(const Eigen::MatrixXd& x) {
//    int rows = (int)x.rows();
//    int cols = (int)x.cols();
//    Eigen::MatrixXd y = x;
//    for (int i = 0; i < rows; i++) {
//        double s = 0;
//        for (int j = 0; j < cols; j++) {
//            y(i, j) = exp(y(i, j));
//            s += y(i, j);
//        }
//        for (int j = 0; j < cols; j++)
//            y(i, j) = y(i, j) / s;
//    }
//    return y;
//}

//Eigen::MatrixXd ActivationFunction::tanh(const Eigen::MatrixXd& x) {
//    int rows = (int)x.rows();
//    int cols = (int)x.cols();
//    Eigen::MatrixXd y = x;
//    for (int i = 0; i < rows; i++) {
//        for (int j = 0; j < cols; j++) {
//            double a = exp(y(i, j));
//            double b = exp(-y(i, j));
//            y(i, j) = (a - b) / (a + b);
//        }
//    }
//    return y;
//}

