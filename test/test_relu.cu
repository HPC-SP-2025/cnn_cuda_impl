#include <iostream>     // std::cout
#include <cstdlib>      // size_t, rand
#include <string>       // std:stoi
#include "../include/cnn_library/layers/relu.h"

int main(int charc, char** argv){

    // Input arguments
    size_t input_size = std::stoi(argv[1]);
    size_t output_size = std::stoi(argv[2]);
    size_t batch_size = std::stoi(argv[3]);
    int device = std::stoi(argv[4]);

    // Create ReLU layer
    ReLU* relu = new ReLU(input_size, output_size, batch_size);
    relu->setDevice(device);

    // Input buffer
    float* input = (float*)malloc(sizeof(float)*input_size);
    for (size_t i=0; i<input_size; i++){
        input[i] = ((float)rand()/RAND_MAX) * 6.0f - 3.0f;
        std::cout << input[i];
    }
    std::cout << endl;

    // Forward pass
    float* output = relu->forward(input);

    // Print results
    for (size_t i=0; i<input_size; i++){
        std::cout << output[i];
    }
    std::cout << endl;

    // Clean-up heap
    delete relu;

    return 0;
}