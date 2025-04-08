#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

# include <vector>

class Sequential
{
public:
    // Constructor
    Sequential(int input_size, int output_size);

    // Set the device ID for the layer
    void setDevice(int device);

    // Forward pass
    void forward(const std::vector<float>& input, std::vector<float>& output);

    // Backward pass
    void backward(const std::vector<float>& d_output, std::vector<float>& d_input);

    // Getters for input and output sizes
    int getInputSize() const;
    int getOutputSize() const;

    // Update Parameters
    void updateParameters(float learning_rate);

    // Get Gradients
    void getGradients(std::vector<float>& gradients);

    // Get Params
    void getParams(std::vector<float>& params);

private:
    int input_size;
    int output_size;
    int device;
};

#endif // SEQUENTIAL_H
