#include "NeuralNetworks.h"

CTensor* NeuralNetworks::operator()(CTensor& x){
	return forward(x);
}

//std::vector<Module::Base *> NeuralNetworks::getMember(){
//	return member;
//}
