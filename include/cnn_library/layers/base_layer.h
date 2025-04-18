#ifndef BASE_LAYER_H
#define BASE_LAYER_H

// Include the headers
#include <vector>
#include <string>
#include <memory>
# include "../layers/base_layer.h"

using namespace std;

class Layer 
{

protected:
    string layer_name; // Name of the layer
    int device_id;
    size_t input_size;
    size_t output_size;
    size_t batch_size;

    // For convolution Layer only
    unsigned int input_height;
    unsigned int output_height;
    unsigned int input_width;
    unsigned int output_width;
    

public:
    // Constructor and Destructor
    Layer();
    virtual ~Layer() = default;


    // -------------------------------------------------------------------
    // COMPULSORY FUNCTIONS
    // -------------------------------------------------------------------

    // Forward and Backward pass
    virtual void forward(float* input, float* output) = 0;
    virtual void backward(float* grad_input, float* grad_output) = 0;

    // Get the metadata for the layer
    virtual size_t getInputSize() = 0;
    virtual size_t getOutputSize() = 0;
    virtual size_t numParams() = 0;
    virtual string getLayerName() = 0;

    // set Device ID for the layer
    virtual void setDevice(int device) = 0;
    virtual int getDevice() = 0;
    




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