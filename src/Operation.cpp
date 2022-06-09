#include "Operation.h"

#include <utility>

CTensor* Operation::Base::createTensor(DoubleTensor& m, Operation::Base* t, const char* name, std::vector<CTensor*> v){
	return new CTensor(m, t, name, std::move(v));
}

std::vector<CTensor*> Operation::Base::getOpCTensor(const CTensor& t){
	return t.opCTensor;
}

CTensor *Operation::Expand::forward(CTensor &a, std::vector<unsigned long> shape) {
    std::vector<CTensor*> v;
    v.push_back(&a);
    DoubleTensor data = a.data->expand(std::move(shape));
    return createTensor(data, new Operation::Expand(), "Expand", v);
}

std::vector<DoubleTensor> Operation::Expand::computeGrad(const CTensor &t, DoubleTensor *grad) {
    if (t.isLeaf) {
        throw(std::exception("叶子节点不能计算梯度"));
    }
    std::vector<DoubleTensor> v;
    CTensor *pre = getOpCTensor(t)[0];
    unsigned long preSize = pre->getSize();
    unsigned int times = t.getSize() / preSize;
    DoubleTensor a(pre->getVectorShape());
    // 列向量
    if(pre->getDim() == 2 && pre->getShape()[0] != 1 && pre->getShape()[1] == 1){
        if(t.getDim() == 2 && t.getShape()[0] == pre->getShape()[0] && t.getShape()[1] != 1){
            for(unsigned long i = 0; i < t.getShape()[0]; i++){
                a[i] = 0;
                for(unsigned long j = 0; j < t.getShape()[1]; j++){
                    a[i] += (*grad)({i, j});
                }
            }
        }
    }
    else{
        for(unsigned long i = 0; i < preSize; i++){
            a[i] = (*grad)[i];
            for(unsigned int n = 1; n < times; n++){
                a[i] += (*grad)[i + n * preSize];
            }
        }
    }
    v.push_back(a);
    return v;
}


CTensor* Operation::Dot::forward(CTensor& a, CTensor& b){
    std::vector<CTensor*> v;
    const unsigned long *aShape = a.getShape();
    const unsigned long *bShape = b.getShape();

    if(a.getDim() == b.getDim()){
        for(unsigned int i = 0; i < a.getDim(); i ++){
            if(aShape[i] != bShape[i]){
                throw(std::exception("向量维度不同，无法点乘"));
            }
        }
        v.push_back(&a);
        v.push_back(&b);
        DoubleTensor data = a.getData() * b.getData();
        return createTensor(data, new Operation::Dot(), "Dot", v);
    }
    else if(a.getDim() > b.getDim()){
        for(unsigned long i = a.getDim() - b.getDim(), j = 0; i < a.getDim(); i++, j++){
            if(aShape[i] != bShape[j]){
                throw(std::exception("向量维度不同，无法点乘"));
            }
        }
        std::vector<unsigned long> shape(aShape, aShape + a.getDim());
        CTensor *expendB = Operation::Expand::forward(b, shape);
        v.push_back(&a);
        v.push_back(expendB);
        DoubleTensor data = a.getData() * expendB->getData();
        return createTensor(data, new Operation::Dot(), "Dot", v);
    }
    else{
        for(unsigned long i = b.getDim() - a.getDim(), j = 0; i < b.getDim(); i++, j++){
            if(bShape[i] != aShape[j]){
                throw(std::exception("向量维度不同，无法点乘"));
            }
        }
        std::vector<unsigned long> shape(bShape, bShape + b.getDim());
        CTensor *expendA = Operation::Expand::forward(a, shape);
        v.push_back(expendA);
        v.push_back(&b);
        DoubleTensor data = expendA->getData() * b.getData();
        return createTensor(data, new Operation::Dot(), "Dot", v);
    }
}

std::vector<DoubleTensor> Operation::Dot::computeGrad(const CTensor& t, DoubleTensor* grad){
	if (t.isLeaf) {
		throw(std::exception("叶子节点不能计算梯度"));
	}
	std::vector<DoubleTensor> v;
	std::vector<CTensor*> pre = getOpCTensor(t);
    v.push_back(pre[1]->getData() * (*grad));
    v.push_back(pre[0]->getData() * (*grad));
	return v;
}

CTensor* Operation::Mul::forward(CTensor& a, CTensor& b){
    if(a.getDim() != 2 || b.getDim() != 2){
        throw(std::exception("矩阵乘法需要张量维度等于2"));
    }
	std::vector<CTensor*> v;
	v.push_back(&a);
	v.push_back(&b);
	DoubleTensor data = a.getData().mm(b.getData());
	return createTensor(data, new Operation::Mul(), "Mul", v);
}

std::vector<DoubleTensor> Operation::Mul::computeGrad(const CTensor& t, DoubleTensor* grad){
	if (t.isLeaf) {
		throw(std::exception("叶子节点不能计算梯度"));
	}
	const std::vector<CTensor*> pre = getOpCTensor(t);
	std::vector<DoubleTensor> v;
    DoubleTensor temp = pre[1]->getData().transpose();
    v.push_back(grad->mm(temp));
    temp = pre[0]->getData().transpose();
    v.push_back(temp.mm(*grad));
//    v.push_back(*grad * pre[1]->getData().transpose());
//    v.push_back(pre[0]->getData().transpose() * (*grad));
	return v;
}


CTensor* Operation::Add::forward(CTensor& a, CTensor& b){
    std::vector<CTensor*> v;
    if(a.getDim() < b.getDim()){
        std::vector<unsigned long> shape(b.getShape(), b.getShape() + b.getDim());
        CTensor* c = Operation::Expand::forward(a, shape);
        v.push_back(c);
        v.push_back(&b);
        DoubleTensor data = c->getData() + b.getData();
        return createTensor(data, new Operation::Add(), "Add", v);
    }
    else if(a.getDim() > b.getDim()) {
        std::vector<unsigned long> shape(a.getShape(), a.getShape() + a.getDim());
        CTensor* c = Operation::Expand::forward(b, shape);
        v.push_back(&a);
        v.push_back(c);
        DoubleTensor data = a.getData() + c->getData();
        return createTensor(data, new Operation::Add(), "Add", v);
    }
    else{
        if(a.getShape()[0] == 1 || b.getShape()[0] == 1){
            if(a.getShape()[0] < b.getShape()[0]){
                std::vector<unsigned long> shape(b.getShape(), b.getShape() + b.getDim());
                CTensor* c = Operation::Expand::forward(a, shape);
                v.push_back(c);
                v.push_back(&b);
                DoubleTensor data = c->getData() + b.getData();
                return createTensor(data, new Operation::Add(), "Add", v);
            }
            else if(a.getShape()[0] > b.getShape()[0]){
                std::vector<unsigned long> shape(a.getShape(), a.getShape() + a.getDim());
                CTensor* c = Operation::Expand::forward(b, shape);
                v.push_back(&a);
                v.push_back(c);
                DoubleTensor data = a.getData() + c->getData();
                return createTensor(data, new Operation::Add(), "Add", v);
            }
        }
        if(a.getShape()[a.getDim() - 1] == 1 || b.getShape()[b.getDim() - 1] == 1){
            if(a.getShape()[a.getDim() - 1] < b.getShape()[b.getDim() - 1]){
                std::vector<unsigned long> shape(b.getShape(), b.getShape() + b.getDim());
                CTensor* c = Operation::Expand::forward(a, shape);
                v.push_back(c);
                v.push_back(&b);
                DoubleTensor data = c->getData() + b.getData();
                return createTensor(data, new Operation::Add(), "Add", v);
            }
            else if(a.getShape()[a.getDim() - 1] > b.getShape()[b.getDim() - 1]){
                std::vector<unsigned long> shape(a.getShape(), a.getShape() + a.getDim());
                CTensor* c = Operation::Expand::forward(b, shape);
                v.push_back(&a);
                v.push_back(c);
                DoubleTensor data = a.getData() + c->getData();
                return createTensor(data, new Operation::Add(), "Add", v);
            }
        }
        for(unsigned long i = 0; i < a.getDim(); i++){
            if(a.getShape()[i] != b.getShape()[i]){
                throw(std::exception("张量维度不符"));
            }
        }
        v.push_back(&a);
        v.push_back(&b);
        DoubleTensor data = a.getData() + b.getData();
        return createTensor(data, new Operation::Add(), "Add", v);
    }
}

CTensor* Operation::Add::forward(CTensor& a, double n){
	std::vector<CTensor*> v;
	v.push_back(&a);
	DoubleTensor temp({1});
	temp[0] = n;
	v.push_back(new CTensor(temp, false, true));
    DoubleTensor data = a.getData() * n;
	return createTensor(data, new Operation::Add(), "Add", v);
}

CTensor* Operation::Add::forward(double n, CTensor& a){
	return Operation::Add::forward(a, n);
}

std::vector<DoubleTensor> Operation::Add::computeGrad(const CTensor& t, DoubleTensor* grad) {
	if (t.isLeaf) {
		throw(std::exception("叶子节点不能计算梯度"));
	}
	std::vector<DoubleTensor> v;
    v.push_back(*grad);
    v.push_back(*grad);
	return v;
}

CTensor* Operation::Power::forward(CTensor& a, double n) {
	std::vector<CTensor*> v;
	v.push_back(&a);
    DoubleTensor temp({1});
	temp({0})= n;
	v.push_back(new CTensor(temp, false, true));
    DoubleTensor data = a.data->pow(n);
	return createTensor(data, new Operation::Power(), "Power", v);
}

std::vector<DoubleTensor> Operation::Power::computeGrad(const CTensor& t, DoubleTensor* grad){
	if (t.isLeaf) {
		throw(std::exception("叶子节点不能计算梯度"));
	}
	std::vector<DoubleTensor> v;
	const std::vector<CTensor*> pre = getOpCTensor(t);
    double n = pre[1]->getData()({0});
    DoubleTensor data = (pre[0]->data->pow(n-1))*n;
	v.push_back(data * (*grad));
	return v;
}

CTensor* Operation::Exp::forward(CTensor& a){
	std::vector<CTensor*> v;
	v.push_back(&a);
	DoubleTensor data = a.getData().exp();
	return createTensor(data, new Operation::Exp(), "Exp", v);
}

std::vector<DoubleTensor> Operation::Exp::computeGrad(const CTensor& t, DoubleTensor* grad){
	std::vector<DoubleTensor> v;
	v.push_back(t.getData() * (*grad));
	return v;
}

// 以e为底
CTensor* Operation::Log::forward(CTensor& a){
	std::vector<CTensor*> v;
	v.push_back(&a);
	DoubleTensor data = a.getData().log();
	return createTensor(data, new Operation::Log(), "Log", v);
}

std::vector<DoubleTensor> Operation::Log::computeGrad(const CTensor& t, DoubleTensor* grad){
	std::vector<DoubleTensor> v;
	const std::vector<CTensor*> pre = getOpCTensor(t);
	DoubleTensor data = pre[0]->getData().pow(-1.0) * (*grad);
	v.push_back(data);
	return v;
}

// ActivationFunction
CTensor* Operation::activationFunction::Sigmoid::forward(CTensor& t){
    DoubleTensor c({1});
    c[0] = -1;
    CTensor *c1 = new CTensor(c, false, true);
	CTensor* t1 = Operation::Dot::forward(t, *c1);
	CTensor* t2 = Operation::Exp::forward(*t1);
	CTensor* t3 = Operation::Add::forward(*t2, 1);
	CTensor* t4 = Operation::Power::forward(*t3, -1);
	return t4;
}

CTensor* Operation::activationFunction::Relu::forward(CTensor& t) {
	std::vector<CTensor*> v;
	v.push_back(&t);
	DoubleTensor data = t.getData();
    for(unsigned long i = 0; i < data.getSize(); i++){
        if(data[i] < 0){
            data[i] = 0;
        }
    }
	return createTensor(data, new Operation::activationFunction::Relu(), "Relu", v);
}

std::vector<DoubleTensor> Operation::activationFunction::Relu::computeGrad(const CTensor& t, DoubleTensor* grad){
	std::vector<DoubleTensor> v;
    DoubleTensor data = t.getData();
    for(unsigned long i = 0; i < data.getSize(); i++){
        if(data[i] == 0){
            grad->operator[](i) = 0;
        }
    }
	v.push_back(*grad);
	return v;
}

CTensor* Operation::activationFunction::Softmax::forward(CTensor &t) {
	std::vector<CTensor*> v;
	v.push_back(&t);
	DoubleTensor data = t.getData();
    unsigned long temp = data.getShape()[data.getDim() - 1];
    for(unsigned long i = 0; i < data.getSize(); i += temp){
        double sum = 0;
        for(unsigned long j = 0; j < temp; j++){
            data[i+j] = exp(data[i+j]);
            sum += data[i+j];
        }
        for(unsigned long j = 0; j < temp; j++){
            data[i+j] = data[i+j] / sum;
        }
    }
	return createTensor(data, new Operation::activationFunction::Softmax(), "Softmax", v);
}

std::vector<DoubleTensor> Operation::activationFunction::Softmax::computeGrad(const CTensor& t, DoubleTensor* grad){
	std::vector<DoubleTensor> v;
	DoubleTensor data(t.getVectorShape());
	DoubleTensor tensorData = t.getData();
    unsigned long temp = data.getShape()[data.getDim() - 1];
    for(unsigned long i = 0; i < data.getSize(); i += temp){
        for(unsigned long j = 0; j < temp; j++){
            double sum = 0;
            for (unsigned long k = 0; k < temp; k++) {
                if(j == k){
                    sum += grad->operator[](i+k) * tensorData[i+j] * (1 - tensorData[i+k]);
                }
                else{
                    sum += -grad->operator[](i+k) * tensorData[i+j] * tensorData[i+k];
                }
            }
            data[i+j] = sum;
        }
    }
	v.push_back(data);
	return v;
}

CTensor *Operation::activationFunction::LogSoftmax::forward(CTensor &t) {
    CTensor* t1 = Operation::activationFunction::Softmax::forward(t);
    CTensor* t2 = Operation::Log::forward(*t1);
    return t2;
}


// LossFunction 
CTensor* Operation::lossFunction::Mean::forward(CTensor& t) {
	DoubleTensor data({1});
    double sum = 0;
    for(unsigned i = 0; i < t.getSize(); i++){
        sum += t[i];
    }
	data[0] = sum/t.getSize();
	std::vector<CTensor*> v;
	v.push_back(&t);
	return createTensor(data, new Operation::lossFunction::Mean(), "Mean", v);
}

std::vector<DoubleTensor> Operation::lossFunction::Mean::computeGrad(const CTensor& t, DoubleTensor* grad){
    DoubleTensor m = getOpCTensor(t)[0]->getData();
	DoubleTensor p(m.getVectorShape());
	double temp = 1.0 / m.getSize();
	for (unsigned long i = 0; i < m.getSize(); i++) {
			p[i] = temp;
	}
	std::vector<DoubleTensor> v;
	v.push_back(p);
	return v;
}

CTensor* Operation::lossFunction::MSELoss::forward(CTensor& t, CTensor& target){
    if(t.getDim() == target.getDim()){
        for(unsigned long i = 0; i < t.getDim(); i ++){
            if(t.getShape()[i] != target.getShape()[i]){
                throw(std::exception("输入张量与目标维度不同"));
            }
        }
    }
    else{
        throw(std::exception("输入张量与目标维度不同"));
    }
	std::vector<CTensor*> v;
	v.push_back(&t);
	v.push_back(&target);
	DoubleTensor data = t.getData() - target.getData();
    data.pow_(2);
    double sum = 0;
    for(unsigned long i = 0; i < data.getSize(); i++){
        sum += data[i];
    }
	DoubleTensor temp({1});
	temp[0] = sum/data.getSize();
	return createTensor(temp, new Operation::lossFunction::MSELoss(), "MSELoss", v);
}

std::vector<DoubleTensor> Operation::lossFunction::MSELoss::computeGrad(const CTensor& t, DoubleTensor* grad){
    std::vector<DoubleTensor> v;
    std::vector<CTensor*> pre = getOpCTensor(t);
	const CTensor target = *pre[1];
	const CTensor temp = *pre[0];
	DoubleTensor data = temp.getData() - target.getData();
	double n = 2.0 /pre[0]->getData().getSize();
	v.push_back(data*n);
	return v;
}

CTensor* Operation::lossFunction::CrossEntropyLoss::forward(CTensor& t, CTensor& target){
	CTensor* t1 = Operation::activationFunction::Softmax::forward(t);
	CTensor* t2 = Operation::Log::forward(*t1);
	CTensor* t3 = Operation::lossFunction::NLLLoss::forward(*t2, target);
	return t3;
}

CTensor* Operation::lossFunction::NLLLoss::forward(CTensor& t, CTensor& target){
    if(t.getDim() > 2){
        throw(std::exception("该函数暂时只支持dim<=2的张量"));
    }
    if (target.getDim() != 1) {
        throw(std::exception("target目标需为一维张量"));
    }
	if (t.getShape()[0] != target.getShape()[0]) {
		throw(std::exception("输入张量与目标维度不同"));
	}
	std::vector<CTensor*> v;
	v.push_back(&t);
	v.push_back(&target);
	double loss = 0;
	for (unsigned long i = 0; i < t.getShape()[0]; i++) {
        loss += t({i, (unsigned long)target[i]});
	}
	DoubleTensor data({1});
	data[0] = - loss/t.getShape()[0];
	return createTensor(data, new  Operation::lossFunction::NLLLoss(), "NLLLoss", v);
}

std::vector<DoubleTensor> Operation::lossFunction::NLLLoss::computeGrad(const CTensor& t, DoubleTensor* grad){
	std::vector<DoubleTensor> v;
	std::vector<CTensor*> pre = getOpCTensor(t);
    DoubleTensor target = pre[1]->getData();
    DoubleTensor data({pre[0]->getShape()[0], pre[0]->getShape()[1]});
	double temp = -1.0 / target.getShape()[0];
	for (unsigned long i = 0; i < target.getSize(); i++) {
		data({i, (unsigned long)target[i]}) = temp;
	}
	v.push_back(data);
	return v;
}