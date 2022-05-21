#pragma once
#include <vector>
#include <Eigen/Dense>
#include "Module.h"
#include "Tensor.h"

class NeuralNetworks{
protected:
	std::vector<Module::Base*> member;

public:
	virtual ~NeuralNetworks() = default;

	virtual Tensor* forward(Tensor& x) = 0;
	Tensor* operator()(Tensor& x);

	std::vector<Module::Base*> getMember() const;
};