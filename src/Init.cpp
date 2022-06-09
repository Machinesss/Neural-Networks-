#include "Init.h"

#include <utility>

unsigned int Init::calculateCorrectFan(CTensor& t, const std::string& mode){
	if (mode != "fanIn" && mode != "fanOut") {
		throw(std::exception("mode只设置为fanIn或fanOut"));
	}
    unsigned int *fan = calculateFanInAndFanOut(t);
    unsigned int temp = mode == "fanIn" ? fan[0] : fan[1];
	delete fan;
	return temp;
}

unsigned int* Init::calculateFanInAndFanOut(CTensor& t){
    unsigned int* a = new unsigned int[2];
	a[0] = t.getShape()[0];
	a[1] = t.getShape()[1];
	return a;
}

void Init::kaimingUniform(CTensor& t, double a, const std::string& mode, const std::string& nonlinearity){
    unsigned int fan = calculateCorrectFan(t, mode);
	double gain = calculateGain(nonlinearity, true, a);
	double std = gain / sqrt(fan);
	double bound = sqrt(3.0) * std;
	uniform(t, -bound, bound);
}

void Init::uniform(CTensor& t, double a, double b){
    std::random_device rd;
    std::default_random_engine seed{rd()};
	std::uniform_real_distribution<double> randomUniform(a, b);

	for (unsigned long i = 0; i < t.getSize(); i++) {
			t[i] = randomUniform(seed);
	}
}

double Init::calculateGain(const std::string& nonlinearity, bool flag, double param){
	if (nonlinearity == "sigmoid") {
		return 1.0;
	}
	if (nonlinearity == "tanh") {
		return 5.0/3;
	}
	if (nonlinearity == "relu") {
		return sqrt(2.0);
	}
	const std::string str[7] = { "linear", "conv1d", "conv2d", "conv3d", "convTranspose1d", "convTranspose2d", "conv_transpose3d" };
	for (int i = 0; i < 7; i++) {
		if (nonlinearity == str[i]) {
			return 1.0;
		}
	}
	if (nonlinearity == "leakyRelu") {
		double negativeSlope = 0.01;
		if (flag) {
			negativeSlope = param;
		}
		return sqrt(2.0 / (1 + pow(negativeSlope, 2)));
	}
	throw(std::exception("nonlinearity 不合法"));
}
