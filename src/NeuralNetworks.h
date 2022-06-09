#pragma once
#include <vector>
#include "Module.h"
#include "CTensor.h"

class NeuralNetworks{
public:
    std::vector<Module::Base*> member;

	virtual ~NeuralNetworks() = default;

	virtual CTensor* forward(CTensor& x) = 0;
    CTensor* operator()(CTensor& x);

//	std::vector<Module::Base *> getMember();
};