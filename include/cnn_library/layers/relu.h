#ifndef RELU_H
#define RELU_H

#include <iostream>
#include <vector>
#include <string>
#include "base_layer.h"

class ReLU : public Layer {

protected:  // TODO remove protected variables
    string layer_name; 
    int device = 0; 
    size_t input_size;
    size_t output_size;
    size_t batch_size;

    // Input buffer
    float* layer_input_ptr;

    // Forward Buffer
    float* host_forward_buffer;
    float* device_forward_buffer;

    // Backward Buffer
    float* host_backward_buffer;
    float* device_backward_buffer;

    // CUDA parameters
    size_t threads_per_block = 1024;

public:

    // Constructor
    ReLU(size_t input_size, size_t output_size, size_t batch_size);
    
    // Destructor
    ~ReLU();

    // Forward pass override
    float* forward(float* input) override;

    // Backward pass override
    void backward(float* grad_input, float* grad_output) override;

    // Set the device ID for the layer
    void setDevice(int device) override;

    // Get device ID for the layer
    int getDevice() override;

    // Get input size
    size_t getInputSize() override;

    // Get output size
    size_t getOutputSize() override;

    // Get layer name
    string getLayerName() override;

    // Get number of parameters
    size_t numParams() override;

private:

    // CPU IMPLEMENTATION
    void forwardCpuReLU(float* input, float* output);
    void backwardCpuReLU(float* grad_input, float* grad_output);

};

#endif  // RELU_H
