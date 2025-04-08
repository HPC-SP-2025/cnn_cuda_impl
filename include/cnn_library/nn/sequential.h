#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

# include <vector>
# include "cnn_library/layers/base_layer.h"

class Sequential
{

private:
    int input_size;
    int output_size;
    int device;
    
public:
    // Constructor
    Sequential(int input_size, int output_size);

    // Destructor
    ~Sequential();

    // Add a layer to the model
    void addLayer(Layer* layer);

    // Set the device ID for the layer
    void setDevice(int device);

    // Forward pass
    void forward(const std::vector<float>& input, std::vector<float>& output);

    // Backward pass
    float backward(const std::vector<float>& d_output, std::vector<float>& d_input, Layer* loss_layer);

    // Getters for input and output sizes
    int getInputSize() const;
    int getOutputSize() const;

    // Update Parameters
    void updateParameters(float learning_rate);

    // Get Gradients
    void getGradients(std::vector<float>& gradients);

    // Get Params
    void getParams(std::vector<float>& params);


};

#endif // SEQUENTIAL_H
