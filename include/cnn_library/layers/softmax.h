#include <iostream>
#include <vector>
#include <string>
#include "base_layer.h"

class Softmax : public Layer {

protected:
    string layer_name; // Name of the layer
    int device = 0; // 0 for CPU, 1 for GPU
    size_t input_size;
    size_t output_size;
    size_t batch_size;

    // Forward Buffer
    float* host_forward_buffer;
    float* device_forward_buffer;

    // Backward Buffer
    float* host_backward_buffer;
    float* device_backward_buffer;

public:
    // Constructor
    Softmax(size_t num_classes, size_t batch_size);

    // Destructor
    ~Softmax();

    // Forward pass override
    float* forward(float* input) override;

    // Backward pass override
    float* backward(float* grad_input) override;

    // Get input size
    size_t getInputSize() override;

    // Get output size
    size_t getOutputSize() override;

    // Number of weights or bias in the layer
    size_t numParams() override;

    // Get name of the layer
    string getLayerName() override;

    // Set the device ID for the layer
    void setDevice(int device) override;

    // Get device state of the layer
    int getDevice() override;

private:

    // CPU IMPLEMENTATION
    void forwardCpuSoftmax(float* input, float* output);
    void backwardCpuSoftmax(float* grad_input, float* grad_output);
    __host__ void forwardGpuSoftmax(float* input, float* output);
    __host__ void backwardGpuSoftmax(float* grad_input, float* grad_output);
};

// CUDA KERNEL IMPLEMENTATION
__global__ void forwardKernelSoftmax(float* input, float* output, size_t num_classes, size_t batch_size);
__global__ void backwardKernelSoftmax(float* grad_input, float* grad_output, size_t num_classes, size_t batch_size);