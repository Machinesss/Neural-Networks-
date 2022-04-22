#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <exception>
#include "LossFunction.h"
#include "ActivationFunction.h"
#include "Adam.h"
#include "NeuralNetworks.h"
#include "Linear.h"
//#include <mat.h>


#include <ctime>
#include <ratio>
#include <chrono>


using namespace Eigen;
using namespace std;

class BPNN : public NeuralNetworks<MatrixXd> {
public:
    BPNN(Module<MatrixXd>* b, double lr = 0.1): NeuralNetworks<>(b, lr){
        //learnRate = lr;
        Linear<>* temp = new Linear<MatrixXd>(5, 2, new sigmoid, new MESLoss);
        MatrixXd w(2, 5);
        MatrixXd bias(1, 2);
        w << -0.0428, -0.3958, -0.3858, -0.4024, -0.4376,
            -0.4165, -0.3388, -0.4034, -0.4195, 0.1960;
        bias << 0.2743, 0.3022;
        temp->setWeight(w);
        temp->setBias(bias);
        v.push_back(temp);
        //v.push_back(new Linear<>(1024, 10, new sigmoid, new MESLoss));
    }
};

//void loadData(MatrixXd & trainData) {
//    MATFile* pmatFile = NULL;
//    mxArray* pMxArray = NULL;
//    pmatFile = matOpen("MNIST.mat", "r");
//    pMxArray = matGetVariable(pmatFile, "trainData");
//    int* matArray = (int*)mxGetData(pMxArray);
//    cout << matArray[0] << endl;
//    int M = mxGetM(pMxArray);
//    int N = mxGetN(pMxArray);
//    trainData = MatrixXd(M, N);
//    //for (int i = 0; i < N; i++)
//    //    for (int j = 0; j < M; j++)
//            // TODO //
//            //trainData.col(i)[j] = matArray[M * i + j];
//
//    cout << trainData.row(1) << endl;
//    matClose(pmatFile);
//    mxFree(matArray);
//}


int main() {
    //MatrixXd trainData;
    //loadData(trainData);

    //SparseXd x(4, 4);
    //for (int i = 0; i < 4; i++) {
    //    x.insert(i, i) = i;
    //}

    MatrixXd  x(4, 3);
    x << 0.7694, 0.6694, 0.7203,
        0.2235, 0.9502, 0.4655,
        0.9314, 0.6533, 0.8914,
        0.8988, 0.3955, 0.3546;
    MatrixXd target(4, 2);
    target << 0.5752, 0.4787,
        0.5782, 0.7536,
        0.1093, 0.4771,
        0.1076, 0.9829;

    Linear<MatrixXd> *linear = new Linear<MatrixXd>(3, 5, new sigmoid);
    MatrixXd w(5, 3);
    MatrixXd b(1, 5);
    w << -0.0484, -0.0198, -0.2165,
        0.1328, -0.3303, -0.1018,
        0.2238, 0.5419, 0.1360,
        -0.1959, 0.0554, -0.0647,
        0.2357, 0.0661, 0.2262;
    b << 0.5599, -0.2397, -0.0204, 0.1328, -0.0038;

    linear->setBias(b);
    linear->setWeight(w);

    BPNN model(linear);
    MatrixXd y(4, 2);
    double loss;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 20000; i++) {
        y = model(x);
        //cout << "y:" << endl;
        //cout << y << endl;
        loss = model.getLoss(target);
        //printf("%d. loss = %lf\n", i + 1, loss);
        model.backward(target);
    }
    auto end = std::chrono::high_resolution_clock::now();

    y = model(x);
    cout << "y:" << endl;
    cout << y << endl;

    std::cout << "in millisecond time:";
    std::chrono::duration<double, std::ratio<1, 1000>> diff = end - start;
    std::cout << "Time is " << diff.count() << " ms\n";
    //MatrixXd y = model(x);
    //cout << y << endl;
    //printf("loss: %lf", model.getLoss(target));
    //// train
    //for (int i = 0; i < 200; i++) {
    //    MatrixXd y = model(x);
    //    printf("%d. ", i);
    //    cout << "output:\n" << y << endl;
    //    try {
    //        cout << "loss: " << model.getLoss(y, target) << endl;
    //    }
    //    catch (const exception& e) {
    //        cout << e.what() << endl;
    //    }
    //    model.backward(1e-1, &target);
    //    printf("\n-------------------------------\n");
    //}    
    return 0;
}
