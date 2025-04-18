#include "../../include/cnn_library/nn/sequential.h"
#include <iostream>

// Constructor
Sequential::Sequential(int input_size, int output_size)
    {
    // Initialization logic if needed
    this ->input_size = input_size;
    this ->output_size = output_size;
    cout << "Sequential model created with input size: " << input_size << " and output size: " << output_size << endl;
}

// Destructor
Sequential::~Sequential() {

}

// Add a layer to the model
void Sequential::addLayer(Layer* layer) 
{
    this->layers.push_back(layer);
}

// Set the device ID for the layer
void Sequential::setDevice(int device) 
{
    this->device = device;

    for (Layer* layer :layers)
    {
        layer->setDevice(device);
    }
    std::cout << "Device set to: " << device << std::endl;  
}



// Forward pass
void Sequential::forward(float* input, float* output) 
{
    for (Layer* layer : layers) 
    {
        // Call the forward method of each layer
        layer->forward(input, output);
        input = output; // Update input for the next layer
    }
    // Final output
    output = input; // Set the final output
}

// Backward pass
float Sequential::backward(float* predicted, float* ground_truth, Layer* loss_layer) 
{
    float loss_value = 0.0f;
    float* loss_ptr = new float[1];
    float* output = nullptr;

    // Forward Pass
    loss_value = loss_layer->forward(predicted, ground_truth);

    // Backward Pass through the loss layer
    loss_layer-> backward(output, loss_value);
    

    // Traverse the layers in reverse order
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) 
    {
        Layer* layer = *it;
        layer->backward(output, input);
        input = output; // Update output for the next layer in reverse
    }

    return loss_value;

}


// Getters for input and output sizes
int Sequential::getInputSize() const {
    return input_size;
}

int Sequential::getOutputSize() const {
    return output_size;
}

// Update Parameters
void Sequential::updateParameters(float learning_rate) 
{
    for (Layer* layer : layers) 
    {
        // Call the update method of each layer
        layer->updateParameters(learning_rate);
    }
}

// Get Gradients
void Sequential::getGradients(std::vector<float>& gradients) {
    // Logic to collect gradients from all layers
}

// Get Params
void Sequential::getParams(std::vector<float>& params) {
    // Logic to collect parameters from all layers
}

void Sequential::saveModel(const string filename) 
{
    // Logic to save the model to a file
    std::cout << "Model saved to: " << filename << std::endl;
}
// Load Model
