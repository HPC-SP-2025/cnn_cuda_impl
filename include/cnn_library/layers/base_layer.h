#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include <vector>
#include <string>
#include <memory>

using namespace std;

class Layer 
{

protected:
    string layer_name; // Name of the layer
    int device_id;
public:
    // Constructor and Destructor
    Layer();
    virtual ~Layer() = default;


    // -------------------------------------------------------------------
    // COMPULSORY FUNCTIONS
    // -------------------------------------------------------------------

    // Forward and Backward pass
    virtual void forward(const std::vector<float>& input, std::vector<float>& output) = 0;
    virtual void backward(const std::vector<float>& grad_output, std::vector<float>& grad_input) = 0;

    // Getters for input and output sizes
    virtual size_t getInputSize() = 0;
    virtual size_t getOutputSize() = 0;

    // set Device ID for the layer
    virtual void setDevice(int device) = 0;




    // -------------------------------------------------------------------
    // OPTIONAL FUNCTIONS
    // -------------------------------------------------------------------

    // Method to update parameters
    virtual void updateParameters(float learning_rate){} // If the layer has parameters else dont override it

    // Initialize the weights and Biases
    virtual void initializeWeights(){}; // If the layer has weights else dont override it
    virtual void initializeBiases(){};  // If the layer has biases else dont override it




private:

    // CUDA KERNEL IMPLEMENTATION
    // __global__ void forwardKernel(){}; // Example of a CUDA kernel function
    // __global__ void backwardKernel(){}; // Example of a CUDA kernel function

    // CPU IMPLEMENTATION
    // void forwardCPU(){}; // Example of a CPU function
    // void backwardCPU(){}; // Example of a CPU function
};


#endif // BASE_LAYER_H