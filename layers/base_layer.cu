

#include <iostream>
#include <vector>

class Layer 
{
    public:

    // Initialize the layer
        Layer(int input_size, int output_size) 
            : input_size(input_size), output_size(output_size) 
            {}

        // Set the the device ID for the layer
        virtual void setDevice(int device) = 0;

        // Forward pass (pure virtual function to be implemented by derived classes)
        virtual void forward(const std::vector<float>& input, std::vector<float>& output) = 0;

        // Backward pass (pure virtual function to be implemented by derived classes)
        virtual void backward(const std::vector<float>& d_output, std::vector<float>& d_input) = 0;

        // Getters for input and output sizes
        int getInputSize() const { return input_size; }
        int getOutputSize() const { return output_size; }

        // Update Parameters (pure virtual function to be implemented by derived classes)
        virtual void updateParameters(float learning_rate) = 0;

    private:
        int input_size;
        int output_size;

        // The CUDA kernels can be implemented in private
};