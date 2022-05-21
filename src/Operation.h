#pragma once

#include "Tensor.h"
#include <math.h>

namespace Operation {
	class Base {
	protected:
		static Tensor* createTensor(const Eigen::MatrixXd& m, Operation::Base* t, const char* name, std::vector<Tensor*> v);
		static std::vector<Tensor*> getOpTensor(const Tensor& t);

		Eigen::MatrixXd dot(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b);
	public:
		virtual ~Base() = default;

		virtual std::vector<Eigen::MatrixXd> computeGrad(const Tensor& t, Eigen::MatrixXd* grad) = 0;
	};

	class Dot :public Base {
	public:
		static Tensor* forward(Tensor& a, Tensor& b);
		std::vector<Eigen::MatrixXd> computeGrad(const Tensor& t, Eigen::MatrixXd* grad) override;
	};

	class Multiplication :public Base {
	public:
		static Tensor* forward(Tensor& a, Tensor& b);
		static Tensor* forward(Tensor& a, double n);
		static Tensor* forward(double n, Tensor& a);
		std::vector<Eigen::MatrixXd> computeGrad(const Tensor& t, Eigen::MatrixXd* grad) override;
	};

	class Add :public Base {
	public:
		static Tensor* forward(Tensor& a, Tensor& b);
		static Tensor* forward(Tensor& a, double n);
		static Tensor* forward(double n, Tensor& a);
		std::vector<Eigen::MatrixXd> computeGrad(const Tensor& t, Eigen::MatrixXd* grad) override;
	};

	class Power :public Base {
	public:
		static Tensor* forward(Tensor& a, double n);
		std::vector<Eigen::MatrixXd> computeGrad(const Tensor& t, Eigen::MatrixXd* grad) override;
	};

	class Exp :public Base {
	public:
		static Tensor* forward(Tensor& a);
		std::vector<Eigen::MatrixXd> computeGrad(const Tensor& t, Eigen::MatrixXd* grad) override;
	};

	class Log :public Base {
	public:
		static Tensor* forward(Tensor & a);
		std::vector<Eigen::MatrixXd> computeGrad(const Tensor & t, Eigen::MatrixXd * grad) override;
	};

	namespace activationFunction {
		class Base :public Operation::Base {

		};

		class Sigmoid :public Operation::activationFunction::Base{
		public:
			static Tensor* forward(Tensor& t);
			//std::vector<Eigen::MatrixXd> computeGrad(const Tensor& t, Eigen::MatrixXd* grad = nullptr) override;
		};

		class Relu :public Operation::activationFunction::Base {
		public:
			static Tensor* forward(Tensor& t);
			std::vector<Eigen::MatrixXd> computeGrad(const Tensor& t, Eigen::MatrixXd* grad = nullptr) override;
		};

		class Softmax :public Operation::activationFunction::Base {
		public:
			static Tensor* forward(Tensor& t);
			std::vector<Eigen::MatrixXd> computeGrad(const Tensor& t, Eigen::MatrixXd* grad = nullptr) override;
		};
	}

	namespace lossFunction {
		class Base : public Operation::Base {
		
		};

		class Mean : public Operation::lossFunction::Base {
		public:
			static Tensor* forward(Tensor& t);
			std::vector<Eigen::MatrixXd> computeGrad(const Tensor& t, Eigen::MatrixXd* grad) override;
		};

		class MSELoss : public Operation::lossFunction::Base {
		public:
			static Tensor* forward(Tensor& t, Tensor &target);
			std::vector<Eigen::MatrixXd> computeGrad(const Tensor& t, Eigen::MatrixXd* grad) override;
		};

		class CrossEntropyLoss :public Operation::lossFunction::Base {
		public:
			static Tensor* forward(Tensor& t, Tensor& target);
		};
	
		class NLLLoss : public Operation::lossFunction::Base {
		public:
			static Tensor* forward(Tensor& t, Tensor& target);
			std::vector<Eigen::MatrixXd> computeGrad(const Tensor& t, Eigen::MatrixXd* grad) override;
		};
	}
};