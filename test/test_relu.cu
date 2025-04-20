#include <iostream>     // std::cout
#include <cstdlib>      // size_t, rand
#include <string>       // std::stoi
#include <ctime>        // time
#include "../include/cnn_library/layers/relu.h"

int main(int argc, char** argv){

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
    if (!input) {
        std::cerr << "Failed to allocate input buffer.\n";
        return 1;
    }
    std::srand(static_cast<unsigned int>(std::time(0)));
    std::cout << "Inputs: ";
    for (size_t i=0; i<input_size*batch_size; i++){
        input[i] = ((float)rand()/RAND_MAX) * 6.0f - 3.0f;
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;

    // Forward pass
    float* output = relu->forward(input);

    // Print results
    std::cout << "Outputs: ";
    for (size_t i=0; i<input_size; i++){
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    // Clean-up heap
    free(input);
    delete relu;

    return 0;
}