#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <exception>

#include "NeuralNetworks.h"
#include "Tensor.h"
#include "Operation.h"
#include "Optimizer.h"
#include "Module.h"

#include <ctime>
#include <fstream>
#include <Windows.h>

using namespace Eigen;
using namespace std;

vector<Tensor*> loadIrisData() {
    vector<Tensor*> dataAndLabel;
    MatrixXd data();
    MatrixXd label;
    const char* trainPath = "../../../../dataset/iris_training.csv";
    const char* testPath = "../../../../dataset/iris_test.csv";

    ifstream inFile(trainPath);
    if (!inFile) {
        throw(exception("打开文件失败"));
    }

    string strLine;
    getline(inFile, strLine); 
    MatrixXd trainData(120, 4);
    MatrixXd trainLabel(120, 1);
    int i = 0;
    while (getline(inFile, strLine)) {
        int j = 0;
        int nPos = strLine.find(',');
        while (nPos > 0) {
            string strTemp = strLine.substr(0, nPos);
            trainData.row(i)[j] = atof(strTemp.c_str());
            j++;
            strLine = strLine.substr(nPos + 1);
            nPos = strLine.find(',');
        }
        trainLabel.row(i)[0] = atof(strLine.c_str());
        i++;
    }
    inFile.close();
    dataAndLabel.push_back(new Tensor(trainData));
    dataAndLabel.push_back(new Tensor(trainLabel));
    inFile.open(testPath);
    if (!inFile) {
        throw(exception("打开文件失败"));
    }
    getline(inFile, strLine);
    MatrixXd testData(30, 4);
    MatrixXd testLabel(30, 1);
    i = 0;
    while (getline(inFile, strLine)) {
        int j = 0;
        int nPos = strLine.find(',');
        while (nPos > 0) {
            string strTemp = strLine.substr(0, nPos);
            testData.row(i)[j] = atof(strTemp.c_str());
            j++;
            strLine = strLine.substr(nPos + 1);
            nPos = strLine.find(',');
        }
        testLabel.row(i)[0] = atof(strLine.c_str());
        i++;
    }
    inFile.close();
    dataAndLabel.push_back(new Tensor(testData));
    dataAndLabel.push_back(new Tensor(testLabel));
    return dataAndLabel;
}

class BPNN : public NeuralNetworks {
public:
    BPNN(){
        Module::Linear* linear1 = new  Module::Linear(4, 10);
        Module::Linear* linear2 = new  Module::Linear(10, 3);

        member.push_back(linear1);
        member.push_back(linear2);
    }

    Tensor* forward(Tensor& x) override {
        Tensor* y = member[0]->forward(x);
        y = Operation::activationFunction::Relu::forward(*y);
        y = member[1]->forward(*y);
        return y;
    }
};

double accuracy(Tensor y, Tensor label) {
    Tensor* t = Operation::activationFunction::Softmax::forward(y);
    MatrixXd p = t->getData();
    double acc = 0;
    for (int i = 0; i < p.rows(); i++) {
        double max = 0;
        int maxIndex = 0;
        for (int j = 0; j < p.cols(); j++) {
            if (p.row(i)[j] > max) {
                max = p.row(i)[j];
                maxIndex = j;
            }
        }
        if (maxIndex == label.getData().row(i)[0]) {
            acc += 1;
        }
    }
    return acc / p.rows();
}

void train(NeuralNetworks* model, Optimizer::Base *opt, Tensor &trainData, Tensor &trainLabel, unsigned int times) {
    Tensor* y = model->forward(trainData);
    Tensor* loss = Operation::lossFunction::CrossEntropyLoss::forward(*y, trainLabel);
    printf("%d. loss = %.7lf, accuracy = %.3lf\n", times, loss->getData().row(0)[0], accuracy(*y, trainLabel));
    loss->backward();
    opt->step();
}

void test(NeuralNetworks* model, Tensor& testData, Tensor& testLabel) {
    Tensor* y = model->forward(testData);
    Tensor* loss = Operation::lossFunction::CrossEntropyLoss::forward(*y, testLabel);
    printf("test loss = %.7lf, accuracy = %.3lf\n", loss->getData().row(0)[0], accuracy(*y, testLabel));
}

int main() {
    vector<Tensor*> data;
    try {
        data = loadIrisData();
    }
    catch (exception e) {
        cout << e.what() << endl;
    }
    Tensor* trainData = data[0];
    Tensor* trainLabel = data[1];
    Tensor* testData = data[2];
    Tensor* testLabel = data[3];

    BPNN model = BPNN();
    Optimizer::SGD optimizer = Optimizer::SGD(model.getMember(), 0.01, 0.9);
    for (int i = 0; i < 200; i++) {
        train(&model, &optimizer, *trainData, *trainLabel, i + 1);
    }
    printf("test result:\n");
    test(&model, *testData, *testLabel);
    return 0;
}
