#pragma once

#include "CTensor.h"
#include <string>
#include <random>  

class Init {
public:
	static unsigned int calculateCorrectFan(CTensor& t, const std::string& mode);

	static unsigned int* calculateFanInAndFanOut(CTensor& t);
	static void kaimingUniform(CTensor& t, double a = 0, const std::string& mode = "fanIn", const std::string& nonlinearity = "leakyRelu");

	static void uniform(CTensor &t, double a, double b);
	static double calculateGain(const std::string& nonlinearity, bool flag = false, double param = 0);
};