#include "Module.h"

std::vector<Tensor*> Module::Base::getParm(){
	return parm;
}

Tensor* Module::Base::operator()(Tensor& x){
	return forward(x);
}

Module::Linear::Linear(unsigned int inputFeature, unsigned int outputFeature, bool needBias): inFeature(inputFeature), outFeature(outputFeature) ,requiresBias(needBias){
	if (inputFeature == 0 || outputFeature == 0) {
		throw(std::exception("输入特征数和输出特征数不能为零"));
	}
	Eigen::MatrixXd data = Eigen::MatrixXd(inputFeature, outputFeature);
	Tensor* w = new Tensor(data, true);
	Init::kaimingUniform(*w, sqrt(5.0));
	parm.push_back(w);
	if (needBias) {
		data = Eigen::MatrixXd(1, outFeature);
		Tensor* bias = new Tensor(data, true);
		int fan_in = Init::calculateFanInAndFanOut(*w)[0];
		double bound = 1 / sqrt(fan_in);
		Init::uniform(*bias, -bound, bound);
		parm.push_back(bias);
	}
}

Tensor* Module::Linear::forward(Tensor& x){
	int rows = (int)x.getData().rows();
	int cols = (int)x.getData().cols();

	if (cols != inFeature)
		throw(std::exception(("当前张量的特征数为" + std::to_string(cols) + "，它应当为" + std::to_string(inFeature)).c_str()));

	Tensor* t1 = Operation::Multiplication::forward(x, *parm[0]);
	if (requiresBias) {
		//Eigen::MatrixXd data(rows, outFeature);
		//Eigen::MatrixXd b = parm[1]->getData();
		//for (int i = 0; i < rows; i++) {
		//	data.row(i) = b;
		//}
		Tensor* t2 = Operation::Add::forward(*t1, *parm[1]);
		return t2;
	}
	return t1;
}

void Module::Linear::setWeight(Eigen::MatrixXd data){
	parm[0]->setData(data);
}

void Module::Linear::setBias(Eigen::MatrixXd data){
	parm[1]->setData(data);
}
