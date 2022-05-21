#pragma once

#include "Module.h"
#include <map>
#include <Eigen/Dense>


namespace Optimizer {
    class Base {
    protected:
        double learnRate;
        std::vector<Module::Base*> nnParm;

    public:
        Base(std::vector<Module::Base*> &parm, double lr = 1e-3);
        virtual ~Base() = default;

        //void zeroGrad();
        virtual void step() = 0;
    };

    class SGD :public Base {
    protected:
        double momentum;
        double dampening;
        double weightDecay;
        bool nesterov;
        std::map<Tensor*, Eigen::MatrixXd> state;

    public:
        SGD(std::vector<Module::Base*>& parm, double lr = 1e-3, double momentum = 0, double dampening = 0, double weightDecay = 0, bool nesterov = false);
        ~SGD();
        void step() override;
    };


}
