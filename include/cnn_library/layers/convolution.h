#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H

#include "base_layer.h"

class Convolution : public Layer {

  private:
    unsigned int input_channels;
    unsigned int output_channels;
    unsigned int kernel_size;
    // unsigned int stride;
    unsigned int padding;
    unsigned int input_height;
    unsigned int output_height;
    unsigned int input_width;
    unsigned int output_width;
    float *cached_input;

  public:
    // Constructor
    Convolution(int batch_size, int input_channels, int height, int width, int output_channels, int kernel_size,
                int padding = 0);

    // Destructor
    ~Convolution();

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

    // Update parameters
    void updateParameters(float learning_rate) override;

    // Initialize weights and biases
    void initializeWeights() override;
    void initializeBiases() override;

  private:
    float *forward_CPU(float *input, float *kernel);
    float *forward_GPU(float *input, float *kernel);

    float *backward_CPU(float *grad_input);
    float *backward_GPU(float *grad_input);
};

#endif