#include <iostream>
#include <vector>
#include <string>
#include "cnn_library/layers/base_layer.h"

class Convolution : public Layer {

private:
    int input_channels;
    int output_channels;
    int kernel_size;
    int stride;
    int padding;
    int input_width;
    int input_height;       
    int output_width;
    int output_height;  

public:

    // Constructor
    Convolution(int input_channels, int output_channels, int kernel_size, int stride, int padding = 0);
    
    // Destructor
    ~Convolution();

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

    // Update parameters
    void updateParameters(float learning_rate) override;

    // Initialize weights and biases
    void initializeWeights() override;
    void initializeBiases() override;

};