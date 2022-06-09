#include <iostream>
#include <exception>

#include "NeuralNetworks.h"
#include "CTensor.h"
#include "Operation.h"
#include "Optimizer.h"
#include "Module.h"

#include <ctime>
#include <fstream>
#include <Windows.h>

using namespace std;

vector<CTensor*> loadIrisData() {
    vector<CTensor*> dataAndLabel;
    const char* trainPath = R"(E:\C++ project\BPNNProject\dataset\iris_training.csv)";
    const char* testPath = R"(E:\C++ project\BPNNProject\dataset\iris_test.csv)";

    ifstream inFile(trainPath);
    if (!inFile) {
        throw(exception("打开文件失败"));
    }
    string strLine;
    getline(inFile, strLine);
    CTensor* trainData = new CTensor({120, 4});
    CTensor* trainLabel = new CTensor({120});
    unsigned int i = 0;
    while (getline(inFile, strLine)) {
        unsigned int j = 0;
        int nPos = strLine.find(',');
        while (nPos > 0) {
            string strTemp = strLine.substr(0, nPos);
            trainData->operator()({i, j}) = atof(strTemp.c_str());
            j++;
            strLine = strLine.substr(nPos + 1);
            nPos = strLine.find(',');
        }
        trainLabel->operator[](i) = atof(strLine.c_str());
        i++;
    }
    inFile.close();
    dataAndLabel.push_back(trainData);
    dataAndLabel.push_back(trainLabel);
    inFile.open(testPath);
    if (!inFile) {
        throw(exception("打开文件失败"));
    }
    getline(inFile, strLine);
    CTensor* testData = new CTensor({30, 4});
    CTensor* testLabel = new CTensor({30});
    i = 0;
    while (getline(inFile, strLine)) {
        unsigned int j = 0;
        int nPos = strLine.find(',');
        while (nPos > 0) {
            string strTemp = strLine.substr(0, nPos);
            testData->operator()({i, j}) = atof(strTemp.c_str());
            j++;
            strLine = strLine.substr(nPos + 1);
            nPos = strLine.find(',');
        }
        testLabel->operator[](i) = atof(strLine.c_str());
        i++;
    }
    inFile.close();
    dataAndLabel.push_back(testData);
    dataAndLabel.push_back(testLabel);
    return dataAndLabel;
}

class BPNN : public NeuralNetworks {
public:
    BPNN(){
        Module::Linear* linear1 = new  Module::Linear(4, 10);
        Module::Linear* linear2 = new  Module::Linear(10, 3);

//        double t1[] = {-0.0419, -0.2861,  0.1178,  0.2041, -0.2076, -0.0479, -0.4892, -0.4690,0.2694,  0.4502,
//                      -0.0171, -0.0882, -0.1696,  0.0573, -0.0177, -0.4425, -0.4657,  0.2192,0.1694, -0.0345,
//                      -0.1875,  0.1938,  0.0479,  0.1959,  0.1150, -0.4313, -0.3788,  0.3067,0.2203,  0.4314,
//                      0.1150,  0.4693, -0.0560,  0.4849, -0.0033, -0.4499, -0.4510,  0.3379,-0.2765,  0.1533};
//        CTensor *w1 = new CTensor(DoubleTensor({4, 10}, t1), true, false);
//        linear1->setWeight(*w1);
//        double t2[] = {0.3914, 0.3988, -0.1045, -0.1454, 0.0752, -0.0213, 0.0782, 0.2536, -0.3907, -0.0229};
//        CTensor *b1 = new CTensor(DoubleTensor({1, 10}, t2), true, false);
//        linear1->setBias(*b1);
//        double t3[] = {-0.2482, -0.2295,  0.3031,
//                       0.3054, -0.3050, -0.1501,
//                       -0.2224,  0.0264, -0.2024,
//                       0.0605,  0.0996,  0.1107,
//                       -0.0864,  0.0722, -0.2262,
//                       0.1797,  0.2921, -0.0765,
//                       0.0011,  0.1419, -0.3128,
//                       -0.0318, -0.1455,  0.0865,
//                       0.2315, -0.2166, -0.1078,
//                       0.2888,  0.2453, -0.1769};
//        CTensor *w2 = new CTensor(DoubleTensor({10, 3}, t3), true, false);
//        linear2->setWeight(*w2);
//        double t4[] = {-0.2011, -0.1113, 0.1502};
//        CTensor *b2 = new CTensor(DoubleTensor({1, 3}, t4), true, false);
//        linear2->setBias(*b2);

        member.push_back(linear1);
        member.push_back(linear2);
    }

    CTensor* forward(CTensor& x) override {
        CTensor* y = member[0]->forward(x);
        y = Operation::activationFunction::Relu::forward(*y);
        y = member[1]->forward(*y);
        y = Operation::activationFunction::LogSoftmax::forward(*y);
        return y;
    }
};

double accuracy(CTensor &y, CTensor& label) {
    double acc = 0;
    for (unsigned int i = 0; i < y.getShape()[0]; i++) {
        double max = y({i, 0});
        unsigned int maxIndex = 0;
        for (unsigned int j = 1; j < y.getShape()[1]; j++) {
//            double temp = y({i, j});
            if (y({i, j}) > max) {
                max = y({i, j});
                maxIndex = j;
            }
        }
        if (maxIndex == label[i]) {
            acc += 1;
        }
    }
    return acc / y.getShape()[0];
}

void train(NeuralNetworks* model, Optimizer::Base *opt, CTensor &trainData, CTensor &trainLabel, unsigned int times) {
    CTensor* y = model->forward(trainData);
    CTensor* loss = Operation::lossFunction::NLLLoss::forward(*y, trainLabel);
    printf("%d. loss = %.7lf, accuracy = %.3lf\n", times, loss->operator[](0), accuracy(*y, trainLabel));
    loss->backward();
    opt->step();
}


void test(NeuralNetworks* model, CTensor& testData, CTensor& testLabel) {
    CTensor* y = model->forward(testData);
    CTensor* loss = Operation::lossFunction::CrossEntropyLoss::forward(*y, testLabel);
    printf("test loss = %.7lf, accuracy = %.3lf\n", loss->operator[](0), accuracy(*y, testLabel));
}

int main() {
    vector<CTensor*> data;
    try {
        data = loadIrisData();
    }
    catch (exception &e) {
        cout << e.what() << endl;
    }
    CTensor* trainData = data[0];
    CTensor* trainLabel = data[1];
    CTensor* testData = data[2];
    CTensor* testLabel = data[3];

    BPNN model = BPNN();
    Optimizer::SGD optimizer(model.member, 0.01, 0.9);
    for (int i = 0; i < 100; i++) {
        train(&model, &optimizer, *trainData, *trainLabel, i + 1);
    }
    printf("test result:\n");
    test(&model, *testData, *testLabel);
    return 0;
}