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

    

public:
    // Constructor
    Linear(int input_size, int output_size);
    
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

};

#endif // LOSS_H