#pragma once
#include "CTensor.h"
#include <vector>
#include "Operation.h"
#include "Init.h"

namespace Module{
    class Base {
    protected:
        std::vector<CTensor*> parm;

    public:
        virtual ~Base() = default;
    
        virtual CTensor* forward(CTensor& x) = 0;
        std::vector<CTensor*> getParm();

        CTensor* operator()(CTensor& x);
    };

    class Linear :public Module::Base {
    protected:
        const unsigned int inFeature;
        const unsigned int outFeature;
        const bool requiresBias;

    public:
        Linear(unsigned int inputFeature, unsigned int outputFeature, bool needBias = true);
        CTensor* forward(CTensor& x) override;
        void setWeight(CTensor &data);
        void setBias(CTensor &data);
    };
}