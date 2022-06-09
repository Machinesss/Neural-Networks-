#include "Module.h"

std::vector<CTensor*> Module::Base::getParm(){
	return parm;
}

CTensor* Module::Base::operator()(CTensor& x){
	return forward(x);
}

Module::Linear::Linear(unsigned int inputFeature, unsigned int outputFeature, bool needBias): inFeature(inputFeature), outFeature(outputFeature) ,requiresBias(needBias){
	if (inputFeature == 0 || outputFeature == 0) {
		throw(std::exception("输入特征数和输出特征数不能为零"));
	}
	CTensor* w = new CTensor({inputFeature, outputFeature}, true);
	Init::kaimingUniform(*w, sqrt(5.0));
	parm.push_back(w);
	if (needBias) {
		CTensor* bias = new CTensor({1, outFeature}, true);
		unsigned int fan_in = Init::calculateFanInAndFanOut(*w)[0];
		double bound = 1 / sqrt(fan_in);
		Init::uniform(*bias, -bound, bound);
		parm.push_back(bias);
	}
}

CTensor* Module::Linear::forward(CTensor& x){
    if(x.getDim() != 2){
        throw(std::exception("输入张量的维度应为2"));
    }
	if (x.getShape()[1] != inFeature){
        throw(std::exception(("当前张量的特征数为" + std::to_string(x.getShape()[1]) + "，它应当为" + std::to_string(inFeature)).c_str()));
    }
	CTensor* t1 = Operation::Mul::forward(x, *parm[0]);
	if (requiresBias) {
		CTensor* t2 = Operation::Add::forward(*t1, *parm[1]);
		return t2;
	}
	return t1;
}

void Module::Linear::setWeight(CTensor &data){
    if(parm[0]->isSameShape(data)){
        delete parm[0];
        parm[0] = &data;
    }
}

void Module::Linear::setBias(CTensor &data){
    if(parm[1]->isSameShape(data)){
        delete parm[1];
        parm[1] = &data;
    }
}
