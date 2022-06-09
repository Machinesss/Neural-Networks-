#pragma once

#include <utility>
#include <cmath>
#include <vector>

class DoubleTensor {
protected:
	double* data;
	unsigned long* shape;
	unsigned long* mark;
	unsigned long size;
    unsigned int dim;

	std::vector<unsigned long> getIndex(unsigned long i);
public:


    DoubleTensor();
	explicit DoubleTensor(std::vector<unsigned long> tensorShape, double* d = nullptr, double defaultValue = 0);
    DoubleTensor(const DoubleTensor& t);
    virtual ~DoubleTensor();
    void clear();

	double& operator()(std::vector<unsigned long> index);
    double& operator[](unsigned long i);
	// 张量对应元素相乘
    DoubleTensor operator*(const DoubleTensor& a);
    void operator*=(const DoubleTensor& a);
    // 张量所有元素乘n
    DoubleTensor operator*(const double& n);
    void operator*=(const double& n);
	// 张量对应元素相除
    DoubleTensor operator/(const DoubleTensor& a);
    // 张量所有元素除n
    DoubleTensor operator/(const double& n);
	// 张量对应元素相加
    DoubleTensor operator+(const DoubleTensor& a);
    void operator+=(const DoubleTensor& a);
    // 张量所有元素加n
    DoubleTensor operator+(const double& n);
    void operator+=(const double& n);
	// 张量对应元素相减
    DoubleTensor operator-(const DoubleTensor& a);
    // 张量所有元素减n
    DoubleTensor operator-(const double& n);
    // 张量所有元素取负
    DoubleTensor operator-();

    DoubleTensor& operator=(const DoubleTensor& a);
	// 矩阵乘法，仅适用于dim=2
    DoubleTensor mm(DoubleTensor& a);
	// 张量每个元素的n次幂，并返回一个新的张量
    DoubleTensor pow(double n);
    // 张量每个元素的n次幂，并返回本身
    DoubleTensor& pow_(double n);
	// 张量每个元素以e为底的对数，并返回一个新的张量
    DoubleTensor log();
    // 张量每个元素以e为底的对数，并返回本身
    DoubleTensor& log_();
	// 张量每个元素计算e^n, 返回一个新的张量
    DoubleTensor exp();
    // 张量每个元素计算e^n, 并返回本身
    DoubleTensor exp_();
    // 将低维张量扩充至高维
    DoubleTensor expand(std::vector<unsigned long> newShape);

    // 矩阵转置，仅适用于dim=2(生成新张量)
    DoubleTensor transpose();

    DoubleTensor slice(unsigned long i);
    DoubleTensor slice(unsigned long i, unsigned long j);

	void display(unsigned int temp = 0, unsigned long begin = 0, unsigned long end = -1, bool flag = false);
	unsigned long getSize() const;
    const unsigned long* getShape();
    std::vector<unsigned long> getVectorShape() const;
    unsigned int getDim() const;
    bool isEmpty() const;
};

