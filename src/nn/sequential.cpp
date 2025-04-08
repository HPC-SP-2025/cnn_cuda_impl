#include "include/cnn_library/nn/sequential.h"
#include <iostream>

// Constructor
Sequential::Sequential(int input_size, int output_size)
    : input_size(input_size), output_size(output_size), device(0) {
    // Initialization logic if needed
}

// Destructor
Sequential::~Sequential() {
    // Cleanup logic if needed
}

// Add a layer to the model
void Sequential::addLayer(Layer* layer) {
    // Logic to add a layer to the model
}

// Set the device ID for the layer
void Sequential::setDevice(int device) {
    this->device = device;
    // Logic to set the device for all layers
}

// Forward pass
void Sequential::forward(const std::vector<float>& input, std::vector<float>& output) {
    // Logic for the forward pass
}

// Backward pass
float Sequential::backward(const std::vector<float>& d_output, std::vector<float>& d_input, Layer* loss_layer) {
    // Logic for the backward pass
    return 0.0f; // Placeholder return value
}

// Getters for input and output sizes
int Sequential::getInputSize() const {
    return input_size;
}

int Sequential::getOutputSize() const {
    return output_size;
}

// Update Parameters
void Sequential::updateParameters(float learning_rate) {
    // Logic to update parameters of all layers
}

// Get Gradients
void Sequential::getGradients(std::vector<float>& gradients) {
    // Logic to collect gradients from all layers
}

// Get Params
void Sequential::getParams(std::vector<float>& params) {
    // Logic to collect parameters from all layers
}
