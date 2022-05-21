#include "Operation.h"

Tensor* Operation::Base::createTensor(const Eigen::MatrixXd& m, Operation::Base* t, const char* name, std::vector<Tensor*> v){
	return new Tensor(m, t, name, v);
}

std::vector<Tensor*> Operation::Base::getOpTensor(const Tensor& t){
	return t.opTensor;
}

Eigen::MatrixXd Operation::Base::dot(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b){
	int rows = (int)a.rows();
	int cols = (int)a.cols();
	if (b.rows() != rows || b.cols() != cols) {
		throw(std::exception("矩阵维度不相同"));
	}
	Eigen::MatrixXd c(rows, cols);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			c.row(i)[j] = a.row(i)[j] * b.row(i)[j];
		}
	}
	return c;
}

// 矩阵与矩阵的点乘 或 向量与向量的点乘
// 矩阵与向量的点乘（将向量扩充为矩阵），此用法需要a表示矩阵，b表示列向量
Tensor* Operation::Dot::forward(Tensor& a, Tensor& b){
	int flag = 0;
	// a为向量
	if (a.getData().cols() == 1 || a.getData().rows() == 1) {
		if (a.getData().cols() != b.getData().cols() || a.getData().rows() != b.getData().rows()) {
			throw(std::exception("向量维度不同，无法点乘"));
		}
		else {
			flag = 1;
		}
	}
	// a为矩阵，b可以为向量或矩阵
	else {
		// 矩阵与行向量的点乘
		if (b.getData().rows() == 1) {
			if (b.getData().cols() != a.getData().cols()) {
				throw(std::exception("矩阵a的维度与向量b不同，无法点乘"));
			}
			else {
				flag = 2;
			}
		}
		else if (b.getData().cols() == 1) {
			if (b.getData().rows() != a.getData().rows()) {
				throw(std::exception("矩阵a的维度与向量b不同，无法点乘"));
			}
			else {
				flag = 3;
			}
		}
		// 矩阵与矩阵的点乘
		else {
			if (a.getData().cols() != b.getData().cols() || a.getData().rows() != b.getData().rows()) {
				throw(std::exception("矩阵维度不同，无法相加"));
			}
			else {
				flag = 1;
			}
		}
	}
	std::vector<Tensor*> v;
	if (flag == 1) {
		v.push_back(&a);
		v.push_back(&b);
		Eigen::MatrixXd data = a.getData();
		for (int i = 0; i < data.rows(); i++) {
			for (int j = 0; j < data.cols(); j++) {
				data.row(i)[j] *= b.getData().row(i)[j];
			}
		}
		return createTensor(data, new Operation::Dot(), "Dot with tensor(matrix)", v);
	}
	if (flag == 2) {
		v.push_back(&a);
		v.push_back(&b);
		Eigen::MatrixXd data = a.getData();
		for (int i = 0; i < data.rows(); i++) {
			for (int j = 0; j < data.cols(); j++) {
				data.row(i)[j] *= b.getData().row(0)[j];
			}
		}
		return createTensor(data, new Operation::Dot(), "Dot with tensor(row vector)", v);
	}
	if (flag == 3) {
		v.push_back(&a);
		v.push_back(&b);
		Eigen::MatrixXd data = a.getData();
		for (int i = 0; i < data.cols(); i++) {
			for (int j = 0; j < data.rows(); j++) {
				data.col(i)[j] *= b.getData().col(0)[j];
			}
		}
		return createTensor(data, new Operation::Dot(), "Dot with tensor(col vector)", v);
	}
}

std::vector<Eigen::MatrixXd> Operation::Dot::computeGrad(const Tensor& t, Eigen::MatrixXd* grad){
	if (t.isLeaf) {
		throw(std::exception("叶子节点不能计算梯度"));
	}
	std::vector<Eigen::MatrixXd> v;
	std::vector<Tensor*> pre = getOpTensor(t);
	if (strcmp(t.getOpName(), "Dot with tensor(matrix)") == 0) {
		v.push_back(dot(pre[1]->getData(), *grad));
		v.push_back(dot(pre[0]->getData(), *grad));
	}
	else if (strcmp(t.getOpName(), "Dot with tensor(row vector)") == 0) {
		Eigen::MatrixXd data1(pre[0]->getData().rows(), pre[0]->getData().rows());
		for (int i = 0; i < data1.rows(); i++) {
			data1.row(i) = pre[1]->getData();
		}
		v.push_back(dot(data1, *grad));

		Eigen::MatrixXd data2 = dot(pre[0]->getData(), *grad);
		Eigen::MatrixXd temp = data2.row(0);
		for (int i = 1; i < data2.rows(); i++) {
			for (int j = 0; j < data2.cols(); j++) {
				temp.row(0)[j] += data2.row(i)[j];
			}
		}
		v.push_back(temp);
	}
	else{
		Eigen::MatrixXd data1(pre[0]->getData().rows(), pre[0]->getData().rows());
		for (int i = 0; i < data1.cols(); i++) {
			data1.col(i) = pre[1]->getData();
		}
		v.push_back(dot(data1, *grad));

		Eigen::MatrixXd data2 = dot(pre[0]->getData(), *grad);
		Eigen::MatrixXd temp = data2.col(0);
		for (int i = 1; i < data2.cols(); i++) {
			for (int j = 0; j < data2.rows(); j++) {
				temp.col(0)[j] += data2.col(i)[j];
			}
		}
		v.push_back(temp);
	}
	return v;
}

Tensor* Operation::Multiplication::forward(Tensor& a, Tensor& b){
	if (a.getData().cols() != b.getData().rows()) {
		throw(std::exception("矩阵维度不符，无法相乘"));
	}
	std::vector<Tensor*> v;
	v.push_back(&a);
	v.push_back(&b);
	Eigen::MatrixXd data = a.getData() * b.getData();
	return createTensor(data, new Operation::Multiplication(), "Mul with tensor", v);
}

Tensor* Operation::Multiplication::forward(Tensor& a, double n){
	std::vector<Tensor*> v;
	v.push_back(&a);
	Eigen::MatrixXd temp(1, 1);
	temp.row(0)[0] = n;
	v.push_back(new Tensor(temp, false, true));
	Eigen::MatrixXd data = a.getData();
	int rows = (int)data.rows();
	int cols = (int)data.cols();
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			data.row(i)[j] = data.row(i)[j] * n;
		}
	}
	return createTensor(data, new Operation::Multiplication(), "Mul with const", v);
}

Tensor* Operation::Multiplication::forward(double n, Tensor& a){
	return Operation::Multiplication::forward(a, n);
}

std::vector<Eigen::MatrixXd> Operation::Multiplication::computeGrad(const Tensor& t, Eigen::MatrixXd* grad){
	if (t.isLeaf) {
		throw(std::exception("叶子节点不能计算梯度"));
	}
	const std::vector<Tensor*> pre = getOpTensor(t);
	std::vector<Eigen::MatrixXd> v;
	if (strcmp(t.getOpName(), "Mul with tensor") == 0) {
		v.push_back(*grad * pre[1]->getData().transpose());
		v.push_back(pre[0]->getData().transpose() * (*grad));
	}
	else {
		Eigen::MatrixXd data = *grad;
		int rows = (int)data.rows();
		int cols = (int)data.cols();
		double n = pre[1]->getData().row(0)[0];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				data.row(i)[j] *= n;
			}
		}
		v.push_back(data);
	}
	return v;
}

// 矩阵与矩阵的加法 或 向量与向量的加法
// 矩阵与向量的加法（将向量扩充为矩阵），此用法需要a表示矩阵，b表示向量
Tensor* Operation::Add::forward(Tensor& a, Tensor& b){
	int flag = 0;
	// a为向量
	if (a.getData().cols() == 1 || a.getData().rows() == 1) {
		if (a.getData().cols() != b.getData().cols() || a.getData().rows() != b.getData().rows()) {
			throw(std::exception("向量维度不同，无法相加"));
		}
		else {
			flag = 1;
		}
	}
	// a为矩阵，b可以为向量或矩阵
	else {
		// 矩阵与行向量的加法
		if (b.getData().rows() == 1) {
			if (b.getData().cols() != a.getData().cols()) {
				throw(std::exception("矩阵a的维度与向量b不同，无法相加"));
			}
			else {
				flag = 2;
			}
		}
		else if (b.getData().cols() == 1) {
			if (b.getData().rows() != a.getData().rows()) {
				throw(std::exception("矩阵a的维度与向量b不同，无法相加"));
			}
			else {
				flag = 3;
			}
		}
		// 矩阵与矩阵的加法
		else {
			if (a.getData().cols() != b.getData().cols() || a.getData().rows() != b.getData().rows()) {
				throw(std::exception("矩阵维度不同，无法相加"));
			}
			else {
				flag = 1;
			}
		}
	}
	std::vector<Tensor*> v;
	if (flag == 1) {
		v.push_back(&a);
		v.push_back(&b);
		Eigen::MatrixXd data = a.getData() + b.getData();
		return createTensor(data, new Operation::Add(), "Add with tensor(matrix)", v);
	}
	if (flag == 2) {
		v.push_back(&a);
		v.push_back(&b);
		Eigen::MatrixXd data = a.getData();
		for (int i = 0; i < data.rows(); i++) {
			data.row(i) += b.getData();
		}
		return createTensor(data, new Operation::Add(), "Add with tensor(row vector)", v);
	}
	if (flag == 3) {
		v.push_back(&a);
		v.push_back(&b);
		Eigen::MatrixXd data = a.getData();
		for (int i = 0; i < data.cols(); i++) {
			data.col(i) += b.getData();
		}
		return createTensor(data, new Operation::Add(), "Add with tensor(col vector)", v);
	}
}

Tensor* Operation::Add::forward(Tensor& a, double n){
	std::vector<Tensor*> v;
	v.push_back(&a);
	Eigen::MatrixXd temp(1, 1);
	temp.row(0)[0] = n;
	v.push_back(new Tensor(temp, false, true));
	Eigen::MatrixXd data = a.getData();
	int rows = (int)data.rows();
	int cols = (int)data.cols();
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			data.row(i)[j] += n;
		}
	}
	return createTensor(data, new Operation::Add(), "Add with const", v);
}

Tensor* Operation::Add::forward(double n, Tensor& a){
	return Operation::Add::forward(a, n);
}

std::vector<Eigen::MatrixXd> Operation::Add::computeGrad(const Tensor& t, Eigen::MatrixXd* grad) {
	if (t.isLeaf) {
		throw(std::exception("叶子节点不能计算梯度"));
	}
	std::vector<Eigen::MatrixXd> v;
	if (strcmp(t.getOpName(), "Add with tensor(matrix)") == 0) {
		v.push_back(*grad);
		v.push_back(*grad);
	}
	else if (strcmp(t.getOpName(), "Add with tensor(row vector)") == 0) {
		v.push_back(*grad);
		Eigen::MatrixXd data = grad->row(0);
		for (int i = 1; i < grad->rows(); i++) {
			for (int j = 0; j < grad->cols(); j++) {
				data.row(0)[j] += grad->row(i)[j];
			}
		}
		v.push_back(data);
	}
	else if (strcmp(t.getOpName(), "Add with tensor(col vector)") == 0) {
		v.push_back(*grad);
		Eigen::MatrixXd data = grad->col(0);
		for (int i = 1; i < grad->cols(); i++) {
			for (int j = 0; j < grad->rows(); j++) {
				data.col(0)[j] += grad->col(i)[j];
			}
		}
		v.push_back(data);
	}
	else {
		v.push_back(*grad);
	}
	return v;
}

Tensor* Operation::Power::forward(Tensor& a, double n) {
	std::vector<Tensor*> v;
	v.push_back(&a);
	Eigen::MatrixXd temp(1, 1);
	temp.row(0)[0] = n;
	v.push_back(new Tensor(temp, false, true));
	Eigen::MatrixXd data = a.getData();
	int rows = (int)data.rows();
	int cols = (int)data.cols();
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			data.row(i)[j] = pow(data.row(i)[j], n);
		}
	}
	return createTensor(data, new Operation::Power(), "Power", v);
}

std::vector<Eigen::MatrixXd> Operation::Power::computeGrad(const Tensor& t, Eigen::MatrixXd* grad){
	if (t.isLeaf) {
		throw(std::exception("叶子节点不能计算梯度"));
	}
	std::vector<Eigen::MatrixXd> v;

	const std::vector<Tensor*> pre = getOpTensor(t);
	Eigen::MatrixXd data = pre[0]->getData();
	int rows = (int)data.rows();
	int cols = (int)data.cols();
	double n = pre[1]->getData().row(0)[0];
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			data.row(i)[j] = n * pow(data.row(i)[j], n-1);
		}
	}
	v.push_back(dot(data, *grad));
	return v;
}

Tensor* Operation::Exp::forward(Tensor& a){
	std::vector<Tensor*> v;
	v.push_back(&a);
	Eigen::MatrixXd data = a.getData();
	int rows = (int)data.rows();
	int cols = (int)data.cols();
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			data.row(i)[j] = exp(data.row(i)[j]);
		}
	}
	return createTensor(data, new Operation::Exp(), "Exp", v);
}

std::vector<Eigen::MatrixXd> Operation::Exp::computeGrad(const Tensor& t, Eigen::MatrixXd* grad){
	std::vector<Eigen::MatrixXd> v;
	v.push_back(dot(t.getData(), *grad));
	return v;
}

// 以e为底
Tensor* Operation::Log::forward(Tensor& a){
	std::vector<Tensor*> v;
	v.push_back(&a);
	Eigen::MatrixXd data = a.getData();
	int rows = (int)data.rows();
	int cols = (int)data.cols();
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			data.row(i)[j] = log(data.row(i)[j]);
		}
	}
	return createTensor(data, new Operation::Log(), "Log", v);
}

std::vector<Eigen::MatrixXd> Operation::Log::computeGrad(const Tensor& t, Eigen::MatrixXd* grad){
	std::vector<Eigen::MatrixXd> v;
	const std::vector<Tensor*> pre = getOpTensor(t);
	Eigen::MatrixXd data = pre[0]->getData();
	int rows = (int)data.rows();
	int cols = (int)data.cols();
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			data.row(i)[j] = (1.0/ data.row(i)[j]) * grad->row(i)[j];
		}
	}
	v.push_back(data);
	return v;
}

// ActivationFunction

Tensor* Operation::activationFunction::Sigmoid::forward(Tensor& t){
	Tensor* t1 = Operation::Multiplication::forward(t, -1);
	Tensor* t2 = Operation::Exp::forward(*t1);
	Tensor* t3 = Operation::Add::forward(*t2, 1);
	Tensor* t4 = Operation::Power::forward(*t3, -1);
	return t4;
}

Tensor* Operation::activationFunction::Relu::forward(Tensor& t) {
	std::vector<Tensor*> v;
	v.push_back(&t);
	Eigen::MatrixXd data = t.getData();
	int rows = (int)data.rows();
	int cols = (int)data.cols();
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (data.row(i)[j] <= 0) {
				data.row(i)[j] = 0;
			}
		}
	}
	return createTensor(data, new Operation::activationFunction::Relu(), "Relu", v);
}

std::vector<Eigen::MatrixXd> Operation::activationFunction::Relu::computeGrad(const Tensor& t, Eigen::MatrixXd* grad){
	std::vector<Eigen::MatrixXd> v;
	Eigen::MatrixXd data = t.getData();
	int rows = (int)grad->rows();
	int cols = (int)grad->cols();
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (data.row(i)[j] == 0) {
				grad->row(i)[j] = 0;
			}
		}
	}
	v.push_back(*grad);
	return v;

}

Tensor* Operation::activationFunction::Softmax::forward(Tensor& t) {
	std::vector<Tensor*> v;
	v.push_back(&t);
	Eigen::MatrixXd data = t.getData();
	int rows = (int)data.rows();
	int cols = (int)data.cols();
	for (int i = 0; i < rows; i++) {
		double sum = 0;
		for (int j = 0; j < cols; j++) {
			data.row(i)[j] = exp(data.row(i)[j]);
			sum += data.row(i)[j];
		}
		data.row(i) = data.row(i) / sum;
	}
	return createTensor(data, new Operation::activationFunction::Softmax(), "Softmax", v);
}

std::vector<Eigen::MatrixXd> Operation::activationFunction::Softmax::computeGrad(const Tensor& t, Eigen::MatrixXd* grad){
	std::vector<Eigen::MatrixXd> v;
	Eigen::MatrixXd data(t.getData().rows(), t.getData().cols());
	Eigen::MatrixXd tensorData = t.getData();
	int rows = (int)data.rows();
	int cols = (int)data.cols();
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			double temp = 0;
			for (int k = 0; k < cols; k++) {
				if (j == k) {
					temp += grad->row(i)[k] * tensorData.row(i)(j) * (1 - tensorData.row(i)(j));
				}
				else {
					temp += grad->row(i)[k] * (-tensorData.row(i)(j) * tensorData.row(i)(k));
				}
			}
			data.row(i)[j] = temp;
		}
	}
	v.push_back(data);
	return v;
}


// LossFunction 
Tensor* Operation::lossFunction::Mean::forward(Tensor& t) {
	Eigen::MatrixXd data(1, 1);
	data.row(0)[0] = t.getData().sum();
	std::vector<Tensor*> v;
	v.push_back(&t);
	return createTensor(data / t.getData().size(), new Operation::lossFunction::Mean(), "Mean", v);
}

std::vector<Eigen::MatrixXd> Operation::lossFunction::Mean::computeGrad(const Tensor& t, Eigen::MatrixXd* grad){
	Tensor pre = *getOpTensor(t)[0];
	Eigen::MatrixXd m = pre.getData();
	int rows = (int)m.rows();
	int cols = (int)m.cols();
	Eigen::MatrixXd p(rows, cols);
	double temp = 1.0 / m.size();
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			p.row(i)[j] = temp;
		}
	}
	std::vector<Eigen::MatrixXd> v;
	v.push_back(p);
	return v;
}

Tensor* Operation::lossFunction::MSELoss::forward(Tensor& t, Tensor& target){
	if (t.getData().cols() != target.getData().cols() || t.getData().rows() != target.getData().rows()) {
		throw(std::exception("输入张量与目标维度不同"));
	}
	std::vector<Tensor*> v;
	v.push_back(&t);
	v.push_back(&target);
	Eigen::MatrixXd data = t.getData() - target.getData();
	for (int i = 0; i < data.rows(); i++) {
		for (int j = 0; j < data.cols(); j++) {
			data.row(i)[j] = pow(data.row(i)[j], 2);
		}
	}
	Eigen::MatrixXd temp(1, 1);
	temp.row(0)[0] = data.sum()/data.size();
	return createTensor(temp, new  Operation::lossFunction::MSELoss(), "MSELoss", v);
}

std::vector<Eigen::MatrixXd> Operation::lossFunction::MSELoss::computeGrad(const Tensor& t, Eigen::MatrixXd* grad){
	std::vector<Tensor*> pre = getOpTensor(t);
	const Tensor target = *pre[1];
	const Tensor temp = *pre[0];
	if (temp.getData().cols() != target.getData().cols() || temp.getData().rows() != target.getData().rows()) {
		throw(std::exception("输入张量与目标维度不同"));
	}
	Eigen::MatrixXd data = temp.getData() - target.getData();
	double n = 2.0 /pre[0]->getData().size();
	std::vector<Eigen::MatrixXd> v;
	v.push_back(data*n);
	return v;
}

Tensor* Operation::lossFunction::CrossEntropyLoss::forward(Tensor& t, Tensor& target){
	if (t.getData().rows() != target.getData().rows()) {
		throw(std::exception("输入张量与目标维度不同"));
	}
	if (target.getData().cols() != 1) {
		throw(std::exception("目标需为列向量"));
	}
	Tensor* t1 = Operation::activationFunction::Softmax::forward(t);
	Tensor* t2 = Operation::Log::forward(*t1);
	Tensor* t3 = Operation::lossFunction::NLLLoss::forward(*t2, target);
	return t3;
}

Tensor* Operation::lossFunction::NLLLoss::forward(Tensor& t, Tensor& target){
	if (t.getData().rows() != target.getData().rows()) {
		throw(std::exception("输入张量与目标维度不同"));
	}
	if (target.getData().cols() != 1) {
		throw(std::exception("目标需为列向量"));
	}
	std::vector<Tensor*> v;
	v.push_back(&t);
	v.push_back(&target);
	double loss = 0;
	for (int i = 0; i < t.getData().rows(); i++) {
		loss += t.getData().row(i)[target.getData().row(i)[0]];
	}
	Eigen::MatrixXd data(1, 1);
	data.row(0)[0] = -loss/ t.getData().rows();
	return createTensor(data, new  Operation::lossFunction::NLLLoss(), "NLLLoss", v);
}

std::vector<Eigen::MatrixXd> Operation::lossFunction::NLLLoss::computeGrad(const Tensor& t, Eigen::MatrixXd* grad){
	std::vector<Eigen::MatrixXd> v;
	std::vector<Tensor*> pre = getOpTensor(t);
	Eigen::MatrixXd target = pre[1]->getData();
	Eigen::MatrixXd data = Eigen::MatrixXd::Zero(pre[0]->getData().rows(), pre[0]->getData().cols());
	double temp = -1.0 / target.rows();
	for (int i = 0; i < target.rows(); i++) {
		data.row(i)[target.row(i)[0]] = temp;
	}
	v.push_back(data);
	return v;
}
