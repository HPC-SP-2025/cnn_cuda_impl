#include "../../include/cnn_library/nn/sequential.h"
#include "../../include/cnn_library/layers/loss.h"
#include <iostream>

// Constructor
Sequential::Sequential(int input_size, int output_size) {
    // Initialization logic if needed
    this->input_size = input_size;
    this->output_size = output_size;
    cout << "Sequential model created with input size: " << input_size << " and output size: " << output_size << endl;
}

// Destructor
Sequential::~Sequential() {}

// Add a layer to the model
void Sequential::addLayer(Layer *layer) { this->layers.push_back(layer); }

// Set the device ID for the layer
void Sequential::setDevice(int device) {
    this->device = device;

    for (Layer *layer : layers) {
        layer->setDevice(device);
    }
    std::cout << "Device set to: " << device << std::endl;
}

// Forward pass
void Sequential::forward(float *input, float *output) {
    float *current_input = input;
    float *current_output = nullptr;

    for (Layer *layer : layers) {

        // Call the forward method of each layer
        layer->forward(current_input, current_output);

        // Update current_input to point to the current_output
        current_input = current_output;
    }

    // Copy the final output to the provided output pointer
    std::copy(current_input, current_input + layers.back()->getOutputSize(), output);

    // Free the memory of the last current_input if it was dynamically allocated
    if (current_input != input) {
        delete[] current_input;
    }
}

// Backward pass
float Sequential::backward(float *predicted, float *ground_truth, Loss *loss_layer) {
    float loss_value = 0.0f;
    float *loss_ptr = new float[1];
    float *output = nullptr;
    float *input = nullptr;

    // Forward Pass
    loss_layer->setTarget(ground_truth);
    loss_layer->forward(predicted, loss_ptr);

    // Backward Pass through the loss layer
    loss_layer->backward(predicted, output);

    // Traverse the layers in reverse order
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        Layer *layer = *it;
        layer->backward(output, input);
        output = input; // Update output for the next layer in reverse
    }

    return loss_value;
}

// Getters for input and output sizes
int Sequential::getInputSize() const { return input_size; }

int Sequential::getOutputSize() const { return output_size; }

// Update Parameters
void Sequential::updateParameters(float learning_rate) {
    for (Layer *layer : layers) {
        // Call the update method of each layer
        layer->updateParameters(learning_rate);
    }
}

void Sequential::summary() {
    std::cout << "Model Summary:" << std::endl;
    for (Layer *layer : layers) {
        std::cout << "Layer: " << layer->getLayerName() << ", Input Size: (" << layer->getInputSize() << ") "
                  << ", Output Size: (" << layer->getOutputSize() << ")"
                  << ", Number of Parameters: " << layer->numParams() << ", Device ID: " << layer->getDevice()
                  << std::endl;
    }
}

// Get Gradients
void Sequential::getGradients(std::vector<float> &gradients) {
    // Logic to collect gradients from all layers
}

// Get Params
void Sequential::getParams(std::vector<float> &params) {
    // Logic to collect parameters from all layers
}

void Sequential::saveModel(const string filename) {
    // Logic to save the model to a file
    std::cout << "Model saved to: " << filename << std::endl;
}
