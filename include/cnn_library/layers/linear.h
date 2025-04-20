#ifndef LINEAR_H
#define LINEAR_H

#include <vector>
# include "../layers/base_layer.h"

class Linear : public Layer {

private:
    int input_size;
    int output_size;

    // Weights and biases as C type arrays
    float* host_weights;
    float* host_biases;
    float* device_weights;
    float* device_biases;

    // Gradient arrays
    float* host_grad_weights;
    float* host_grad_biases;
    float* device_grad_weights;
    float* device_grad_biases;

    // Intermediate arrays for forward and backward pass
    float* host_buffer;
    float* device_buffer;

    // Cached input for backward pass
    float* cached_input;
    

public:
    // Constructor
    Linear(size_t input_size, size_t output_size, size_t batch_size = 1);
    
    // Destructor
    ~Linear();

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

    // CPU Forward Pass
    void forwardCPU(float* input, float* output);
    // CPU Backward Pass
    void backwardCPU(float* grad_input, float* grad_output);
    // GPU Forward Pass
    void forwardGPU(float* input, float* output);
    // GPU Backward Pass
    void backwardGPU(float* grad_input, float* grad_output);
};

#endif // LOSS_H