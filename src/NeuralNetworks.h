#pragma once
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "Module.h"

template <typename T = Eigen::MatrixXd>
class NeuralNetworks{
protected:
	Module<T>* begin;
	std::vector<Module<Eigen::MatrixXd> *> v;
	double learnRate;
	Eigen::MatrixXd output;
	T input;

public:
	NeuralNetworks(Module<T> *b, double lr = 1e-3);
	virtual ~NeuralNetworks();
	void setBeign(Module<T>* b);


	Eigen::MatrixXd forward(const T& x);
	Eigen::MatrixXd operator()(const T& x);
	void backward(const Eigen::MatrixXd& target);
	double getLoss(const Eigen::MatrixXd& target);
};


template<typename T>
NeuralNetworks<T>::NeuralNetworks(Module<T>* b, double lr) {
	static_assert(std::is_same_v<T, Eigen::MatrixXd> || std::is_same_v<T, Eigen::SparseMatrix<double>>, "类模板参数T应为SparseMatrix<double>或MatrixXd");
	this->begin = b;
	learnRate = lr;
}

template<typename T>
NeuralNetworks<T>::~NeuralNetworks() {
	delete this->begin;
	this->begin = nullptr;
}

template<typename T>
void NeuralNetworks<T>::setBeign(Module<T>* b) {
	if (b) {
		this->begin = b;
	}
	else {
		throw(std::exception("形参指针为空"));
	}
}

template <typename T>
Eigen::MatrixXd NeuralNetworks<T>::forward(const T& x) {
	if (this->begin == nullptr) {
		throw(std::exception("该网络为空"));
	}
	input = x;
	Eigen::MatrixXd temp = this->begin->forward(x);
	if (v.empty()) {
		output = temp;
		return temp;
	}
	for (std::vector<Module<Eigen::MatrixXd>*>::iterator it = v.begin(); it != v.end(); it++) {
		temp = (*it)->forward(temp);
	}
	output = temp;
	return temp;
}

template <typename T>
Eigen::MatrixXd NeuralNetworks<T>::operator()(const T& x) {
	return forward(x);
}
template <typename T>
double NeuralNetworks<T>::getLoss(const Eigen::MatrixXd& target) {
	if ((v.empty() && !this->begin->haveLossFun()) || (!v.empty() && !(*v.back()).haveLossFun())) {
		throw(std::exception("该网络最后一层没有损失函数"));
	}
	if (v.empty() && this->begin->haveLossFun()) {
		return this->begin->getLoss(target);
	}
	if (!v.empty() && (*v.back()).haveLossFun()) {
		return (*v.back()).getLoss(target);
	}
}

// TODO
template <typename T>
void NeuralNetworks<T>::backward(const Eigen::MatrixXd& target) {
	if ((v.empty() && !this->begin->haveLossFun()) || (!v.empty() && !(*v.back()).haveLossFun())) {
		throw(std::exception("该网络最后一层没有损失函数"));
	}
	if (v.empty()) {
		this->begin->backward(this->begin->getLossFunction()->getPd(output, target), learnRate);
	}
	else {
		Eigen::MatrixXd pd = v.back()->getLossFunction()->getPd(output, target);
		pd = v.back()->backward(pd, learnRate);
		for (std::vector<Module<Eigen::MatrixXd>*>::iterator it = v.end() - 1; it != v.begin(); it--) {
			pd = (*it)->backward(pd, learnRate);
		}
		this->begin->backward(pd, learnRate);
	}
}