#include <iostream>
#include <vector>
#include <string>
#include "cnn_library/layers/base_layer.h"

class ReLU : public Layer {

private:
    int input_channels;
    int output_channels;
    int input_width;
    int input_height;       
    int output_width;
    int output_height;  

public:

    // Constructor
    ReLU();
    
    // Destructor
    ~ReLU();

    // Forward pass override
    void forward(const std::vector<float>& input, std::vector<float>& output) override;

    // Backward pass override
    void backward(const std::vector<float>& grad_output, std::vector<float>& grad_input) override;

    // Set the device ID for the layer
    void setDevice(int device) override;

    // Get input size
    size_t getInputSize() override;

    // Get output size
    size_t getOutputSize() override;

private:

    // CUDA KERNEL IMPLEMENTATION
    // __global__ void forwardKernelReLU(){}; // Example of a CUDA kernel function
    // __global__ void backwardKernelReLU(){}; // Example of a CUDA kernel function

    // CPU IMPLEMENTATION
    // void forwardCpuReLU(){}; // Example of a CPU function
    // void backwardCpuReLU(){}; // Example of a CPU function

};