#include "Init.h"

int Init::calculateCorrectFan(Tensor& t, std::string mode){
	if (mode != "fanIn" && mode != "fanOut") {
		throw(std::exception("mode只设置为fanIn或fanOut"));
	}
	int *fan = calculateFanInAndFanOut(t);
	int temp = mode == "fanIn" ? fan[0] : fan[1];
	delete fan;
	return temp;
}

int* Init::calculateFanInAndFanOut(Tensor& t){
	int* a = new int[2];
	a[0] = t.getData().rows();
	a[1] = t.getData().cols();
	return a;
}

void Init::kaimingUniform(Tensor& t, double a, std::string mode, std::string nonlinearity){
	int fan = calculateCorrectFan(t, mode);
	double gain = calculateGain(nonlinearity, a);
	double std = gain / sqrt(fan);
	double bound = sqrt(3.0) * std;
	uniform(t, -bound, bound);
}

void Init::uniform(Tensor& t, double a, double b){
	std::default_random_engine seed(time(NULL));
	std::uniform_real_distribution<double> randomUniform(a, b);

	Eigen::MatrixXd data = t.getData();
	for (int i = 0; i < data.rows(); i++) {
		for (int j = 0; j < data.cols(); j++) {
			data.row(i)[j] = randomUniform(seed);
		}
	}
	t.setData(data);
}

double Init::calculateGain(std::string nonlinearity, bool flag, double param){
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
	return 0.0;
}
