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
    const char* trainPath = R"..\..\..\dataset\iris_training.csv)";
    const char* testPath = R"(..\..\..\dataset\iris_test.csv)";

    ifstream inFile(trainPath);
    if (!inFile) {
        throw(exception("打开文件失败，请检查文件路径是否存在文件"));
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
