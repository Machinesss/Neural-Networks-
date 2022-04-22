#pragma once

#include "Module.h"
#include <vector>

// TODO
class Optimizer {
public:
    //params
    virtual void setParams(const std::vector<Eigen::MatrixXd>& p) final;
};