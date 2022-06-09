#pragma once

#include "CTensor.h"
#include <cmath>

namespace Operation {
	class Base {
	protected:
		static CTensor* createTensor(DoubleTensor& m, Operation::Base* t, const char* name, std::vector<CTensor*> v);
		static std::vector<CTensor*> getOpCTensor(const CTensor& t);
	public:
		virtual ~Base() = default;

		virtual std::vector<DoubleTensor> computeGrad(const CTensor& t, DoubleTensor* grad) = 0;
	};

    class Expand:public Base{
    public:
        static CTensor* forward(CTensor& a, std::vector<unsigned long> shape);
        std::vector<DoubleTensor> computeGrad(const CTensor& t, DoubleTensor* grad) override;
    };

    class Dot :public Base {
	public:
		static CTensor* forward(CTensor& a, CTensor& b);
		std::vector<DoubleTensor> computeGrad(const CTensor& t, DoubleTensor* grad) override;
	};

	class Mul :public Base {
	public:
		static CTensor* forward(CTensor& a, CTensor& b);
		std::vector<DoubleTensor> computeGrad(const CTensor& t, DoubleTensor* grad) override;
	};

	class Add :public Base {
	public:
		static CTensor* forward(CTensor& a, CTensor& b);
		static CTensor* forward(CTensor& a, double n);
		static CTensor* forward(double n, CTensor& a);
		std::vector<DoubleTensor> computeGrad(const CTensor& t, DoubleTensor* grad) override;
	};

	class Power :public Base {
	public:
		static CTensor* forward(CTensor& a, double n);
		std::vector<DoubleTensor> computeGrad(const CTensor& t, DoubleTensor* grad) override;
	};

	class Exp :public Base {
	public:
		static CTensor* forward(CTensor& a);
		std::vector<DoubleTensor> computeGrad(const CTensor& t, DoubleTensor* grad) override;
	};

	class Log :public Base {
	public:
		static CTensor* forward(CTensor & a);
		std::vector<DoubleTensor> computeGrad(const CTensor & t, DoubleTensor * grad) override;
	};

	namespace activationFunction {
		class Base :public Operation::Base {

		};

		class Sigmoid :public Operation::activationFunction::Base{
		public:
			static CTensor* forward(CTensor& t);
		};

		class Relu :public Operation::activationFunction::Base {
		public:
			static CTensor* forward(CTensor& t);
			std::vector<DoubleTensor> computeGrad(const CTensor& t, DoubleTensor* grad) override;
		};

		class Softmax :public Operation::activationFunction::Base {
		public:
			static CTensor* forward(CTensor &t);
			std::vector<DoubleTensor> computeGrad(const CTensor& t, DoubleTensor* grad) override;
		};

        class LogSoftmax :public Operation::activationFunction::Base {
        public:
            static CTensor* forward(CTensor &t);
        };
	}

	namespace lossFunction {
		class Base : public Operation::Base {
		
		};

		class Mean : public Operation::lossFunction::Base {
		public:
			static CTensor* forward(CTensor& t);
			std::vector<DoubleTensor> computeGrad(const CTensor& t, DoubleTensor* grad) override;
		};

		class MSELoss : public Operation::lossFunction::Base {
		public:
			static CTensor* forward(CTensor& t, CTensor &target);
			std::vector<DoubleTensor> computeGrad(const CTensor& t, DoubleTensor* grad) override;
		};

        // 暂时只支持矩阵
		class CrossEntropyLoss :public Operation::lossFunction::Base {
		public:
			static CTensor* forward(CTensor& t, CTensor& target);
		};
        // 暂时只支持矩阵
		class NLLLoss : public Operation::lossFunction::Base {
		public:
			static CTensor* forward(CTensor& t, CTensor& target);
			std::vector<DoubleTensor> computeGrad(const CTensor& t, DoubleTensor* grad) override;
		};
	}
};