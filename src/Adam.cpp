#include "Adam.h"

Adam::Adam() {
    learnRate = 1e-3;
    betas[0] = 0.9;
    betas[1] = 0.999;
    eps = 1e-8;
    weight_decay = 0;
    timestep = 0;
}

Adam::Adam(double lr, const double* b, double e, double wd) {
    if (lr < 0)
        throw(std::exception("learnRate should be greater or equal than 0"));
    if (b[0] < 0 || b[0] >= 1.0)
        throw(std::exception("betas[0] should be smaller than 1.0 and greater or equal than 0"));
    if (b[1] < 0 || b[1] >= 1.0)
        throw(std::exception("betas[1] should be smaller than 1.0 and greater or equal than 0"));
    if (e < 0)
        throw(std::exception("eps should be greater or equal than 0"));
    if (wd < 0)
        throw(std::exception("weight_decay should be greater or equal than 0"));
    learnRate = lr;
    betas[0] = b[0];
    betas[1] = b[1];
    eps = e;
    weight_decay = wd;
    timestep = 0;
}

void Adam::step(double loss) {

}