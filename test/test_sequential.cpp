#include <iostream>
#include "../include/cnn_library/nn/sequential.h"
#include "../include/cnn_library/layers/relu.h"
// #include "../include/cnn_library/layers/softmax.h"

using namespace std;

int main(){


    // Create a Sequential model
    Sequential* model = new Sequential(784, 10); // Example input size of 784 (28x28 image flattened) and output size of 10 (number of classes)
    cout << "Model created with input size: " << model->getInputSize() << " and output size: " << model->getOutputSize() << endl;
    cout << "Adding layers..." << endl;
    // Add layers to the model
    

    return 0;
}


