# include <iostream>
# include <vector>

// This file helps to create the computation graph for the model


class Sequential
{
public:

    // Constructor
    Sequential(int input_size, int output_size)
        : input_size(input_size), output_size(output_size)
    {
        // Initialize the layer
    }
    // Set the device ID for the layer
    void setDevice(int device)
    {
        // Set the device ID for the layer
        this->device = device;
    }
    // Forward pass
    void forward(const std::vector<float>& input, std::vector<float>& output)
    {
        // Forward pass implementation
        // This will be implemented in derived classes
    }
    // Backward pass
    void backward(const std::vector<float>& d_output, std::vector<float>& d_input)
    {
        // Backward pass implementation
        // This will be implemented in derived classes
    }
    // Getters for input and output sizes
    int getInputSize() const { return input_size; }
    int getOutputSize() const { return output_size; }
    // Update Parameters
    void updateParameters(float learning_rate)
    {
        // Update parameters implementation
        // This will be implemented in derived classes
    }
    // Get Gradients
    void getGradients(std::vector<float>& gradients)
    {
        // Get gradients implementation
        // This will be implemented in derived classes
    }
    // Get Params
    void getParams(std::vector<float>& params)
    {
        // Get parameters implementation
        // This will be implemented in derived classes
    }
}