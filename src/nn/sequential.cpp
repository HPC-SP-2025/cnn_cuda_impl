#include "../../include/cnn_library/nn/sequential.h"
#include "../../include/cnn_library/layers/loss.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

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
float* Sequential::forward(float *input) {
    float *current_input = input;
    float *current_output = nullptr;

    // // Print the pointers for debugging
    // std::cout << "Input pointer: " << input << std::endl;
    // std::cout << "Current input pointer: " << current_input << std::endl;
    // std::cout << "Current output pointer: " << current_output << std::endl;

    for (Layer *layer : layers) 
    {
        current_output = layer->forward(current_input);
        current_input = current_output;
    }
    return current_output; // Return the final output
}

// Backward pass
float Sequential::backward(float *predicted, float *ground_truth, Loss *loss_layer) {
    float loss_value = 0.0f;
    float *loss_ptr;
    float *output_grad = nullptr;
    float *input_grad = nullptr;

    
    

    

    // Forward Pass through Loss Layer
    loss_layer->setTarget(ground_truth);
    loss_ptr = loss_layer->forward(predicted);
    loss_value = loss_ptr[0];
    cout << "Loss Value: " << loss_value << std::endl;





    // Backward Pass through the loss layer
    input_grad = loss_layer->backward(predicted);



    // Traverse the layers in reverse order
    for (int i = layers.size() - 1; i >= 0; i--) 
    {
        Layer *layer = layers[i];
        output_grad = layer->backward(input_grad);
        input_grad = output_grad; // Update output for the next layer in reverse
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


void Sequential::loadModel(const string filename) 
{
    std::ifstream file(filename);  // Open the file for reading

    if (!file.is_open()) {  // Check if file is successfully opened
        std::cerr << "Could not open the file!" << std::endl;
        return;
    }

    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) 
    {  
        std::istringstream iss(line);
        std::vector<float> values;
        float value;

        while (iss >> value) {
            values.push_back(value);
        }

        if (line_number < layers.size()) {
            layers[line_number]->setParameters(values);
        } 
        else 
        {
            std::cerr << "Error: More weights in the file than layers in the model!" << std::endl;
            break;
        }
        line_number++;

    
    }
    file.close();  // Close the file after reading

    std::cout << "Model loaded from: " << filename << std::endl;





}


