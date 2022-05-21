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
        //MatrixXd w1(10, 4);
        //w1 << -0.0419, -0.0171, -0.1875, 0.1150,
        //    -0.2861, -0.0882, 0.1938, 0.4693,
        //    0.1178, -0.1696, 0.0479, -0.0560,
        //    0.2041, 0.0573, 0.1959, 0.4849,
        //    -0.2076, -0.0177, 0.1150, -0.0033,
        //    -0.0479, -0.4425, -0.4313, -0.4499,
        //    -0.4892, -0.4657, -0.3788, -0.4510,
        //    -0.4690, 0.2192, 0.3067, 0.3379,
        //    0.2694, 0.1694, 0.2203, -0.2765,
        //    0.4502, -0.0345, 0.4314, 0.1533;
        //MatrixXd bias1(1, 10);
        //bias1 << 0.3914, 0.3988, -0.1045, -0.1454, 0.0752, -0.0213, 0.0782, 0.2536, -0.3907, -0.0229;
        //linear1->setWeight(w1.transpose());
        //linear1->setBias(bias1);
        //MatrixXd w2(3, 10);
        //MatrixXd bias2(1, 3);
        //w2 << -0.2482, 0.3054, -0.2224, 0.0605, -0.0864, 0.1797, 0.0011, -0.0318, 0.2315, 0.2888,
        //    -0.2295, -0.3050, 0.0264, 0.0996, 0.0722, 0.2921, 0.1419, -0.1455, -0.2166, 0.2453,
        //    0.3031, -0.1501, -0.2024, 0.1107, -0.2262, -0.0765, -0.3128, 0.0865, -0.1078, -0.1769;
        //bias2 << -0.2011, -0.1113, 0.1502;
        //linear2->setWeight(w2.transpose());
        //linear2->setBias(bias2);

        member.push_back(linear1);
        member.push_back(linear2);
    }

    Tensor* forward(Tensor& x) override {
        Tensor* y = member[0]->forward(x);
        y = Operation::activationFunction::Relu::forward(*y);
        y = member[1]->forward(*y);
        return y;
        //Tensor* z1 = member[0]->forward(x);
        //Tensor* y1 = Operation::activationFunction::Relu::forward(*z1);
        //Tensor* z2 = member[1]->forward(*y1);
        //return z2;
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
    //cout << "output: " << endl;
    //cout << model(x)->getData() << endl;
    //cout << "target: " << endl;
    //cout << target.getData() << endl;

    //MatrixXd a(1, 5);
    //a << 0.5615, 0.1774, 0.8147, 0.3295, 0.2319;
    //Tensor x(a, true);
    ////MatrixXd b(2, 2);
    ////b << 0.2139, 0.4118,
    ////    0.6938, 0.9693;
    ////Tensor w(b, true);
    //Tensor *y = Operation::activationFunction::Softmax::forward(x);
    //y->requiresGrad = true;
    //Tensor* loss = Operation::lossFunction::Mean::forward(*y);
    //cout << y->getData() << endl;
    //cout << loss->getData() << endl;
    //loss->backward();
    //printf("stop\n");
    //cout << x.getGrad() << endl;
    return 0;
}