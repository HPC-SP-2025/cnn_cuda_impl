#include <iostream>
#include <vector>
#include <string>
#include "cnn_library/layers/base_layer.h"

class ReLU : public Layer {

protected:  // TODO remove protected variables
    string layer_name; // Name of the layer
    int device = 0; // 0 for CPU, 1 for GPU
    size_t input_size;
    size_t output_size;
    size_t batch_size;
    size_t threads_per_block;
    float* layer_input_ptr;

    // Forward Buffer
    float* host_forward_buffer;
    float* device_forward_buffer;

    // Backward Buffer
    float* host_backward_buffer;
    float* device_backward_buffer;

    // CUDA parameters
    int threads_per_block;

public:

    // Constructor
    ReLU(size_t input_size, size_t output_size, size_t batch_size);
    
    // Destructor
    ~ReLU();

    // Forward pass override
    void forward(float* input, float* output) override;

    // Backward pass override
    void backward(float* grad_input, float* grad_output, float* layer_input_ptr) override;

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
    void backwardCpuReLU(float* grad_input, float* grad_output, float* layer_input_ptr){};

};