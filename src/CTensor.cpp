#include "CTensor.h"

#include <utility>
#include "Operation.h"

//#include <utility>

void CTensor::clear(){
	delete data;
	data = nullptr;
	if (requiresGrad) {
		delete grad;
		grad = nullptr;
	}
	if (!isLeaf) {
		opCTensor.clear();
		delete op;
		op = nullptr;
	}
	delete this;
}

CTensor::CTensor(DoubleTensor& m, Operation::Base* t, const char *name, std::vector<CTensor*> v): isLeaf(false), opName(name), isConst(false){
    data = new DoubleTensor(m);
	op = t;
	opCTensor = std::move(v);
	requiresGrad = false;
	grad = nullptr;
}

void CTensor::computeGrad(CTensor&t, DoubleTensor &pre_grad){
	if (t.requiresGrad) {
		*(t.grad) = pre_grad;
	}
	if(!t.isLeaf){
		std::vector<DoubleTensor> tGrad = t.op->computeGrad(t, &pre_grad);
		for (unsigned int i = 0; i < tGrad.size(); i++) {
			computeGrad(*t.opCTensor[i], tGrad[i]);
		}
		t.clear();
	}
}

CTensor::CTensor(std::vector<unsigned long> shape, bool needGrad, bool isC): isLeaf(true), opName(nullptr), requiresGrad(needGrad), isConst(isC){
	data = new DoubleTensor(shape);
	if (needGrad) {
		grad = new DoubleTensor(shape);
	}
	else {
		grad = nullptr;
	}
	op = nullptr;
}

CTensor::CTensor(DoubleTensor d, bool needGrad, bool isC): isLeaf(true), opName(nullptr), requiresGrad(needGrad), isConst(isC){
    data = new DoubleTensor(d);
    if (needGrad) {
        std::vector<unsigned long> shape(data->getShape(), data->getShape() + data->getDim());
        grad = new DoubleTensor(shape);
    }
    else {
        grad = nullptr;
    }
    op = nullptr;
}

CTensor::CTensor(const CTensor& t):isLeaf(t.isLeaf), opName(t.opName), isConst(t.isConst), requiresGrad(t.requiresGrad){
	data = new DoubleTensor(*t.data);
	if (requiresGrad) {
		grad = new DoubleTensor(*t.grad);
	}
	else {
		grad = nullptr;
	}
	opCTensor = t.opCTensor;
	op = t.op;
}

CTensor::~CTensor(){
	
}

CTensor CTensor::backward(){
	if (data->getSize() != 1 || isConst) {
		throw(std::exception("只有非常数标量可以调用backward"));
	}
	computeGrad(*opCTensor[0], op->computeGrad(*this, nullptr)[0]);
	return *this;
}

DoubleTensor& CTensor::getData() const{
	return *data;
}

DoubleTensor CTensor::getGrad() const{
	if (requiresGrad) {
		return *grad;
	}
	throw(std::exception("该节点没有保存梯度"));
}

const char* CTensor::getOpName() const{
	if (!isLeaf) {
		return opName;
	}
	throw(std::exception("该节点为叶子节点， 不存在运算名"));
}

const unsigned long *CTensor::getShape() const {
    return data->getShape();
}

unsigned int CTensor::getDim() const {
    return data->getDim();
}

unsigned long CTensor::getSize() const {
    return data->getSize();
}

double &CTensor::operator()(std::vector<unsigned long> index) {
    return data->operator()({std::move(index)});
}

double &CTensor::operator[](unsigned long index) {
    if(index >= data->getSize()){
        throw(std::exception("超出下标索引"));
    }
    return data->operator[](index);
}

std::vector<unsigned long> CTensor::getVectorShape() const {
    return {getShape(), getShape()+getDim()};
}

bool CTensor::isSameShape(CTensor &a) const {
    if(data->getDim() != a.getDim()){
        return false;
    }
    for(unsigned long i = 0; i < data->getDim(); i++){
        if(data->getShape()[i] != a.getShape()[i]){
            return false;
        }
    }
    return true;
}

CTensor& CTensor::operator=(DoubleTensor d) {
    if(d.getDim() != data->getDim()){
        throw(std::exception("张量维度不同，无法赋值"));
    }
    for(unsigned long i = 0; i < data->getDim(); i++){
        if(data->getShape()[i] != d.getShape()[i]){
            throw(std::exception("张量维度不同，无法赋值"));
        }
    }
    *data = d;
    return *this;
}

void CTensor::display() const {
    data->display();
}
