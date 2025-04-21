#ifndef LINEAR_H
#define LINEAR_H

#include <vector>
#include <random>
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
    float* forward(float* input) override;

    // Backward pass override
    float* backward(float* grad_input) override;

    // Set the device ID for the layer
    void setDevice(int device) override;

    // Get input size
    size_t getInputSize() override;

    // Get output size
    size_t getOutputSize() override;

    // Number of weights
    size_t numParams() override;

    // Name of the layer
    std::string getLayerName() override;

    //Get device
    int getDevice() override;

    // Initialize weights
    void initializeWeights() override;
    
    // Initialize biases
    void initializeBiases() override;

    // Update weights
    void updateParameters(float learning_rate) override;

    // Get and set weights and biases
    void setWeights(float* weights);
    void setBiases(float* biases);
    void getWeights(float* weights);
    void getBiases(float* biases);

private:

    // CPU Forward Pass
    // Returns the pointer to the output
    float* forwardCPU(float* input, float* output);
    // CPU Backward Pass
    float* backwardCPU(float* grad_input);
    // GPU Forward Pass
    float* forwardGPU(float* input, float* output);
    // GPU Backward Pass
    float* backwardGPU(float* grad_input);
};

#endif // LINEAR_H
