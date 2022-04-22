#pragma once
#include <exception>
#include "Optimizer.h"
#include "Module.h"

// TODO
class Adam : public Optimizer {
private:
    double learnRate;
    double betas[2] = {};
    double eps;
    double weight_decay;
    int timestep;

public:
    Adam();
    Adam(double lr, const double* b, double e, double wd);
    void step(double loss);
};