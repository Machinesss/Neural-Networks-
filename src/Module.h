#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "Optimizer.h"
#include "ActivationFunction.h"
#include "LossFunction.h"


template <typename T = Eigen::MatrixXd>
class Module {
protected:
    ActivationFunction* actfun;
    LossFunction* lossfun;
    Eigen::MatrixXd output;
    T input; 

public:
    Module(ActivationFunction* af = nullptr, LossFunction* lf = nullptr);
    virtual ~Module();
    // ǰ�򴫲�
    virtual Eigen::MatrixXd forward(const T& x) = 0;
    Eigen::MatrixXd operator()(const T& x);
    // ���򴫲�
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd pd, double learnRate = 1e-3) = 0;


    ActivationFunction* getActivationFunction();
    LossFunction* getLossFunction();
    void setActivationFunction(ActivationFunction* fun);
    void setLossFunction(LossFunction* fun);
    double getLoss(const Eigen::MatrixXd& target);
    double getLoss(const Eigen::MatrixXd& y, const Eigen::MatrixXd& target);
    bool haveActFun();
    bool haveLossFun();
};


template<typename T>
Module<T>::Module(ActivationFunction* af, LossFunction* lf) {
    static_assert(std::is_same_v<T, Eigen::MatrixXd> || std::is_same_v<T, Eigen::SparseMatrix<double>>, "��ģ�����TӦΪSparseMatrix<double>��MatrixXd");
    actfun = af;
    lossfun = lf;
}

template<typename T>
Module<T>::~Module() {
    delete(actfun);
    actfun = nullptr;
    delete(lossfun);
    lossfun = nullptr;
}

template<typename T>
Eigen::MatrixXd Module<T>::operator()(const T& x) {
    return forward(x);
}

template<typename T>
inline ActivationFunction* Module<T>::getActivationFunction(){
    return actfun;
}

template<typename T>
inline LossFunction* Module<T>::getLossFunction(){
    return lossfun;
}



template<typename T>
void Module<T>::setActivationFunction(ActivationFunction* fun) {
    if (fun == nullptr)
        throw(std::exception("���ܰ󶨿�ָ����Ϊ�����"));
    actfun = fun;
}

template<typename T>
bool Module<T>::haveActFun() {
    return actfun != nullptr;
}

template<typename T>
void Module<T>::setLossFunction(LossFunction* fun) {
    if (fun == nullptr)
        throw(std::exception("���ܰ󶨿�ָ����Ϊ��ʧ����"));
    lossfun = fun;
}

template<typename T>
double Module<T>::getLoss(const Eigen::MatrixXd& target) {
    if (lossfun == nullptr) {
        throw(std::exception("��ģ��û�а���ʧ����"));
    }
    if (actfun != nullptr) {
        return lossfun->forward(actfun->forward(output), target);
    }
    return lossfun->forward(output, target);
}

template<typename T>
double Module<T>::getLoss(const Eigen::MatrixXd& y, const Eigen::MatrixXd& target) {
    if (lossfun == nullptr) {
        throw(std::exception("��ģ��û�а���ʧ������"));
    }
    return lossfun->forward(y, target);
}

template<typename T>
bool Module<T>::haveLossFun() {
    return lossfun != nullptr;
}