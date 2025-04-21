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
    std::cout << "FORWARD PASS\n";
    float* input = (float*)malloc(sizeof(float)*input_size*batch_size);
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
    for (size_t i=0; i<input_size*batch_size; i++){
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    // Input gradient buffer
    std::cout << "BACKWARD PASS\n";
    float* grad_input = (float*)malloc(sizeof(float)*output_size*batch_size);
    if (!grad_input) {
        std::cerr << "Failed to allocate input gradient buffer.\n";
        return 1;
    }
    std::cout << "Gradient Inputs: ";
    for (size_t i=0; i<output_size*batch_size; i++){
        grad_input[i] = ((float)rand()/RAND_MAX) * 6.0f - 3.0f;
        std::cout << grad_input[i] << " ";
    }
    std::cout << std::endl;

    // Backward pass
    float* grad_output = relu->backward(grad_input);

    // Print output gradients
    std::cout << "Gradient Outputs: ";
    for (size_t i=0; i<output_size*batch_size; i++){
        std::cout << grad_output[i] << " ";
    }
    std::cout << std::endl;

    // Clean-up heap
    free(input);
    delete relu;

    return 0;
}
