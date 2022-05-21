#pragma once

#include <Eigen/Dense>
#include <vector>

namespace Operation {
	class Base;
	class Multiplication;
	class Add;
}


class Tensor {
private:
	void clear();

protected:
	friend class Operation::Base;

	Eigen::MatrixXd data;
	Eigen::MatrixXd grad;
	const char* opName;
	Operation::Base* op;
	std::vector<Tensor*> opTensor;
	
	Tensor(const Eigen::MatrixXd& m, Operation::Base* t, const char* name, std::vector<Tensor*> v);

	void computeGrad(Tensor& t, Eigen::MatrixXd grad);
public:
	const bool isLeaf;
	const bool isConst;
	bool requiresGrad;
	//const unsigned int dim;

	Tensor(const Eigen::MatrixXd& m, bool needGrad = false, bool isC = false);
	Tensor(const Tensor& t);
	virtual ~Tensor();
	
	//Tensor operator*(Tensor& b) const;
	//Tensor operator+(Tensor& b) const;

	Tensor backward();

	Eigen::MatrixXd getData() const;
	Tensor setData(const Eigen::MatrixXd &m);
	Eigen::MatrixXd getGrad() const;
	const char* getOpName() const;
};