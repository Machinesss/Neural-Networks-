#include "Optimizer.h"

Optimizer::Base::Base(std::vector<Module::Base*>& parm, double lr) :learnRate(lr) {
	if (lr < 0.0) {
		throw(std::exception("学习率lr需大于零"));
	}
	nnParm = parm;
}

Optimizer::SGD::SGD(std::vector<Module::Base*>& parm, double lr, double momentum, double dampening, double weightDecay, bool nesterov) :Base(parm, lr){
	if (momentum < 0.0) {
		throw(std::exception("动量momentum需大于零"));
	}
	if (weightDecay < 0.0) {
		throw(std::exception("weightDecay需大于零"));
	}
	if (nesterov && (momentum <= 0 || dampening != 0)) {
		throw(std::exception("nesterov需要设置动量momentum且dampening为零"));
	}
	this->momentum = momentum;
	this->dampening = dampening;
	this->weightDecay = weightDecay;
	this->nesterov = nesterov;
}

Optimizer::SGD::~SGD(){

}

void Optimizer::SGD::step(){
	for (unsigned int i = 0; i < nnParm.size(); i++) {
		std::vector<CTensor*> moduleParm = nnParm[i]->getParm();
		for (unsigned int j = 0; j < moduleParm.size(); j++) {
			DoubleTensor dp = moduleParm[j]->getGrad();
			if (weightDecay != 0) {
				dp += moduleParm[j]->getData() * weightDecay;
			}
			if (momentum != 0) {
				if (state.find(moduleParm[j]) == state.end()) {
                    state.insert(std::pair<CTensor*, DoubleTensor>(moduleParm[j], DoubleTensor(dp)));
				}
				else {
                    state.find(moduleParm[j])->second = state.find(moduleParm[j])->second * momentum + dp * (1 - dampening);
				}
				if (nesterov) {
					dp += state.find(moduleParm[j])->second * momentum;
				}
				else{
					dp = state.find(moduleParm[j])->second;
				}
			}
			dp *= -learnRate;
            moduleParm[j]->operator=(moduleParm[j]->getData() + dp);
		}
	}
}
