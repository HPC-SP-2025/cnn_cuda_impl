#include <iostream>     // std::cout
#include <cstdlib>      // size_t, rand
#include <string>       // std:stoi
#include "../include/cnn_library/layers/softmax.h"

int main(int argc, char** argv) {

    // Input arguments
    size_t num_classes = std::stoi(argv[1]);
    size_t batch_size = std::stoi(argv[2]);
    int device = std::stoi(argv[3]);

    // Create Softmax layer
    Softmax* softmax = new Softmax(num_classes, batch_size);
    softmax->setDevice(device);

    // Input buffer
    float* input = (float*)malloc(sizeof(float) * num_classes * batch_size);
    for (size_t i = 0; i < num_classes * batch_size; i++){
        input[i] = ((float)rand() / RAND_MAX) * 6.0f - 3.0f;
        std::cout << input[i];
    }
    std::cout << endl;

    // Forward pass
    float* output = softmax->forward(input);

    // Print results
    for (size_t i = 0; i < num_classes * batch_size; i++){
        std::cout << output[i];
    }
    std::cout << endl;

    // Clean-up heap
    delete softmax;

    return 0;
}
