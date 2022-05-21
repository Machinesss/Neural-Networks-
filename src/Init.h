#pragma once

#include "Tensor.h"
#include <string>
#include <time.h>  
#include <random>  

class Init {
public:
	static int calculateCorrectFan(Tensor& t, std::string mode);

	static int* calculateFanInAndFanOut(Tensor& t);
	static void kaimingUniform(Tensor& t, double a = 0, std::string mode = "fanIn", std::string nonlinearity = "leakyRelu");

	static void uniform(Tensor &t, double a, double b);
	static double calculateGain(std::string nonlinearity, bool flag = false, double param = 0);
};