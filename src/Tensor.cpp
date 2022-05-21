#include "Tensor.h"
#include "Operation.h"

#include <iostream>

void Tensor::clear(){
	delete op;
	op = nullptr;
	if (!isLeaf) {
		opTensor.clear();
	}
	delete this;
}

Tensor::Tensor(const Eigen::MatrixXd& m, Operation::Base* t, const char *name, std::vector<Tensor*> v): isLeaf(false), opName(name), isConst(false){
	data = m;
	op = t;
	opTensor = v;
	requiresGrad = false;
}

void Tensor::computeGrad(Tensor &t, Eigen::MatrixXd pre_grad){
	if (t.requiresGrad) {
		t.grad = pre_grad;
	}
	if(!t.isLeaf){
		std::vector<Eigen::MatrixXd> tGrad = t.op->computeGrad(t, &pre_grad);
		for (int i = 0; i < tGrad.size(); i++) {
			computeGrad(*t.opTensor[i], tGrad[i]);
		}
		t.clear();
	}
}

Tensor::Tensor(const Eigen::MatrixXd& m, bool needGrad, bool isC): isLeaf(true), opName(nullptr), requiresGrad(needGrad), isConst(isC){
	data = m;
	op = nullptr;
}

Tensor::Tensor(const Tensor& t):isLeaf(t.isLeaf), opName(t.opName), isConst(t.isConst) {
	data = t.data;
	opTensor = t.opTensor;
	grad = t.grad;
	op = t.op;
	requiresGrad = t.requiresGrad;
}

Tensor::~Tensor(){
	//if (!isLeaf) {
	//	delete op;
	//	op = nullptr;
	//}
}

//Tensor Tensor::operator*(Tensor& b) const{
//	return Operation::Multiplication::forward(*this, b);
//}
//
//Tensor Tensor::operator+(Tensor& b) const{
//	return Operation::Add::forward(*this, b);
//}

Tensor Tensor::backward(){
	if (data.size() != 1 || isConst) {
		throw(std::exception("只有非常数标量可以调用backward"));
	}
	computeGrad(*opTensor[0], op->computeGrad(*this, nullptr)[0]);
	return *this;
}

Eigen::MatrixXd Tensor::getData() const{
	return data;
}

Tensor Tensor::setData(const Eigen::MatrixXd& m){
	data = m;
	return *this;
}

Eigen::MatrixXd Tensor::getGrad() const{
	if (requiresGrad) {
		return grad;
	}
	throw(std::exception("该节点没有保存梯度"));
}

const char* Tensor::getOpName() const{
	if (!isLeaf) {
		return opName;
	}
	throw(std::exception("该节点为叶子节点， 不存在运算名"));
}
