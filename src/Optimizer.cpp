#include "Optimizer.h"

Optimizer::Base::Base(std::vector<Module::Base*>& parm, double lr) :learnRate(lr) {
	if (lr < 0.0) {
		throw(std::exception("ѧϰ��lr�������"));
	}
	nnParm = parm;
}

Optimizer::SGD::SGD(std::vector<Module::Base*>& parm, double lr, double momentum, double dampening, double weightDecay, bool nesterov) :Base(parm, lr){
	if (momentum < 0.0) {
		throw(std::exception("����momentum�������"));
	}
	if (weightDecay < 0.0) {
		throw(std::exception("weightDecay�������"));
	}
	if (nesterov && (momentum <= 0 || dampening != 0)) {
		throw(std::exception("nesterov��Ҫ���ö���momentum��dampeningΪ��"));
	}
	this->momentum = momentum;
	this->dampening = dampening;
	this->weightDecay = weightDecay;
	this->nesterov = nesterov;
}

Optimizer::SGD::~SGD(){

}

void Optimizer::SGD::step(){
	for (int i = 0; i < nnParm.size(); i++) {
		std::vector<Tensor*> moduleParm = nnParm[i]->getParm();
		for (int j = 0; j < moduleParm.size(); j++) {
			Eigen::MatrixXd dp = moduleParm[j]->getGrad();
			if (weightDecay != 0) {
				dp += moduleParm[j]->getData() * weightDecay;
			}
			if (momentum != 0) {
				if (state.find(moduleParm[j]) == state.end()) {
					state[moduleParm[j]] = dp;
				}
				else {
					state[moduleParm[j]] = state[moduleParm[j]] * momentum + dp * (1 - dampening);
				}
				if (nesterov) {
					dp += state[moduleParm[j]] * momentum;
				}
				else{
					dp = state[moduleParm[j]];
				}
			}
			dp = -learnRate * dp;
			moduleParm[j]->setData(moduleParm[j]->getData() + dp);
		}
	}
}
