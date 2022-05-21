#pragma once
#include "Tensor.h"
#include <vector>
#include <Eigen/Dense>
#include "Operation.h"
#include "Init.h"

namespace Module{
    class Base {
    protected:
        //Tensor output;
        //Tensor input;
        std::vector<Tensor*> parm;

    public:
        virtual ~Base() = default;
    
        virtual Tensor* forward(Tensor& x) = 0;
        std::vector<Tensor*> getParm();

        Tensor* operator()(Tensor& x);
    };

    class Linear :public Module::Base {
    protected:
        const unsigned int inFeature;
        const unsigned int outFeature;
        const bool requiresBias;

    public:
        Linear(unsigned int inputFeature, unsigned int outputFeature, bool needBias = true);
        Tensor* forward(Tensor& x) override;
        void setWeight(Eigen::MatrixXd data);
        void setBias(Eigen::MatrixXd data);
    };
}