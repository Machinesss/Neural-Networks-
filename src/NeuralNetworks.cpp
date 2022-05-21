#include "NeuralNetworks.h"

Tensor* NeuralNetworks::operator()(Tensor& x){
	return forward(x);
}

std::vector<Module::Base*> NeuralNetworks::getMember() const{
	return member;
}
