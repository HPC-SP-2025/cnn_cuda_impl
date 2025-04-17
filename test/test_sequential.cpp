#include <iostream>
#include "../include/cnn_library/nn/sequential.h"

int main(){


    // Create a Sequential model
    Sequential model(784, 10); // Example input size of 784 (28x28 image flattened) and output size of 10 (number of classes)

    return 0;
}


