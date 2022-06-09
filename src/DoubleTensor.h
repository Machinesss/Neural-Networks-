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
	// ������ӦԪ�����
    DoubleTensor operator*(const DoubleTensor& a);
    void operator*=(const DoubleTensor& a);
    // ��������Ԫ�س�n
    DoubleTensor operator*(const double& n);
    void operator*=(const double& n);
	// ������ӦԪ�����
    DoubleTensor operator/(const DoubleTensor& a);
    // ��������Ԫ�س�n
    DoubleTensor operator/(const double& n);
	// ������ӦԪ�����
    DoubleTensor operator+(const DoubleTensor& a);
    void operator+=(const DoubleTensor& a);
    // ��������Ԫ�ؼ�n
    DoubleTensor operator+(const double& n);
    void operator+=(const double& n);
	// ������ӦԪ�����
    DoubleTensor operator-(const DoubleTensor& a);
    // ��������Ԫ�ؼ�n
    DoubleTensor operator-(const double& n);
    // ��������Ԫ��ȡ��
    DoubleTensor operator-();

    DoubleTensor& operator=(const DoubleTensor& a);
	// ����˷�����������dim=2
    DoubleTensor mm(DoubleTensor& a);
	// ����ÿ��Ԫ�ص�n���ݣ�������һ���µ�����
    DoubleTensor pow(double n);
    // ����ÿ��Ԫ�ص�n���ݣ������ر���
    DoubleTensor& pow_(double n);
	// ����ÿ��Ԫ����eΪ�׵Ķ�����������һ���µ�����
    DoubleTensor log();
    // ����ÿ��Ԫ����eΪ�׵Ķ����������ر���
    DoubleTensor& log_();
	// ����ÿ��Ԫ�ؼ���e^n, ����һ���µ�����
    DoubleTensor exp();
    // ����ÿ��Ԫ�ؼ���e^n, �����ر���
    DoubleTensor exp_();
    // ����ά������������ά
    DoubleTensor expand(std::vector<unsigned long> newShape);

    // ����ת�ã���������dim=2(����������)
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

