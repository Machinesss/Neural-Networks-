#pragma once

#include "DoubleTensor.h"
#include <vector>

namespace Operation {
	class Base;
}

class CTensor {
private:
	void clear();

protected:
	friend class Operation::Base;

	DoubleTensor* grad;
	const char* opName;
	Operation::Base* op;
	std::vector<CTensor*> opCTensor;

	CTensor(DoubleTensor& m, Operation::Base* t, const char* name, std::vector<CTensor*> v);

	void computeGrad(CTensor& t, DoubleTensor &grad);

public:
    DoubleTensor* data;
	const bool isLeaf;
	const bool isConst;
	bool requiresGrad;

	explicit CTensor(std::vector<unsigned long> sharp, bool needGrad = false, bool isC = false);
    explicit CTensor(DoubleTensor d, bool needGrad = false, bool isC = false);
	CTensor(const CTensor& t);
	virtual ~CTensor();

	CTensor backward();

    double& operator()(std::vector<unsigned long> index);
    double& operator[](unsigned long index);
    CTensor& operator=(DoubleTensor d);
    DoubleTensor& getData() const;
    DoubleTensor getGrad() const;
	const char* getOpName() const;
    const unsigned long* getShape() const;
    std::vector<unsigned long> getVectorShape() const;
    unsigned int getDim() const;
    unsigned long getSize() const;
    bool isSameShape(CTensor &a) const;
    void display() const;
};