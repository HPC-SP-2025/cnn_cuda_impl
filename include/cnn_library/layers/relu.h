#include <iostream>
#include <vector>
#include <string>
#include "cnn_library/layers/base_layer.h"

class ReLU : public Layer {

private:
    int input_size;
    int output_size;
    int device; 

    // Output buffer
    float* host_output_buffer;
    float* device_output_buffer;

    // CUDA parameters
    int blocks;
    int threads_per_block;

public:

    // Constructor
    ReLU(int input_size, int output_size);
    
    // Destructor
    ~ReLU();

    // Forward pass override
    void forward(float* input, float* output) override;

    // Backward pass override
    void backward(float* grad_input, float* grad_output) override;

    // Set the device ID for the layer
    void setDevice(int device) override;

    // Get input size
    size_t getInputSize() override;

    // Get output size
    size_t getOutputSize() override;

private:

    // // CUDA KERNEL IMPLEMENTATION
    // __global__ void forwardKernelReLU(float* input, float* output){};
    // __global__ void backwardKernelReLU(){};

    // CPU IMPLEMENTATION
    void forwardCpuReLU(float* input, float* output){};
    void backwardCpuReLU(){};

};