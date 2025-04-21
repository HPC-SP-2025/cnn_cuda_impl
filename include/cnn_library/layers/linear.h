#ifndef LINEAR_H
#define LINEAR_H

#include "../layers/base_layer.h"
#include <random>
#include <vector>

class Linear : public Layer {

  private:
    // Cached input for backward pass
    float *cached_input;

  public:
    // Constructor
    Linear(size_t input_size, size_t output_size, size_t batch_size = 1);

    // Destructor
    ~Linear();

    // Forward pass override
    float *forward(float *input) override;

    // Backward pass override
    float *backward(float *grad_input) override;

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

    // Get device
    int getDevice() override;

    // Initialize weights
    void initializeWeights() override;

    // Initialize biases
    void initializeBiases() override;

    // Update weights
    void updateParameters(float learning_rate) override;

    // Get and set weights and biases
    void setWeights(float *weights);
    void setBiases(float *biases);
    void getWeights(float *weights);
    void getBiases(float *biases);

  private:
    // CPU Forward Pass
    // Returns the pointer to the output
    float *forwardCPU(float *input);
    // CPU Backward Pass
    float *backwardCPU(float *grad_input);
    // GPU Forward Pass
    float *forwardGPU(float *input);
    // GPU Backward Pass
    float *backwardGPU(float *grad_input);
};

__global__ void forward_kernel(float *input, float *weights, float *biases, float *output, size_t input_size, size_t output_size, size_t batch_size);

__global__ void backward_input_kernel(float *grad_output, float *weights, float *grad_input, size_t input_size, size_t output_size, size_t batch_size);

__global__ void backward_weight_kernel(float *input, float *grad_output, float *grad_weights, size_t input_size, size_t output_size, size_t batch_size);

__global__ void backward_bias_kernel(float *grad_output, float *grad_biases, size_t output_size, size_t batch_size);

__global__ void update_parameters_kernel(float* weights, float* grad_weights,
    float* biases, float* grad_biases,
    float learning_rate,
    size_t weight_size, size_t bias_size);

#endif // LINEAR_H
