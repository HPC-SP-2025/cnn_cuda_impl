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
    int set_device = 0; // 0 for CPU, 1 for GPU

    // For convolution Layer only
    unsigned int input_height;
    unsigned int output_height;
    unsigned int input_width;
    unsigned int output_width;

    // Weights and Biases
    float* host_weights;
    float* host_biases;
    float* device_weights;
    float* device_biases;

    // Gradients
    float* host_grad_weights;
    float* host_grad_biases;
    float* device_grad_weights;
    float* device_grad_biases;

    // Forward Buffer
    float* host_forward_buffer;
    float* device_forward_buffer;

    // Backward Buffer
    float* host_backward_buffer;
    float* device_backward_buffer;
    
    

public:
    // Constructor:
    /* Constructor is responsible for inilializing the weights and biases, gradients,
    forward_buffer and the backward_buffer for the layer on the CPU*/
    Layer();
    virtual ~Layer() = default;



    // -------------------------------------------------------------------
    // COMPULSORY FUNCTIONS
    // -------------------------------------------------------------------

    // FORWARD FUNCTION
    /* the forward function gets the pointer(*input) of the output of the previous
    memory, it will perform forward operation and then store output in the
    memory that it created in the constructor and save the pointer to 
    that memory for the next layer in the second arguement(*output)*/
    virtual void forward(float* input, float* output) = 0;

    // BACKWARD FUNCTION
    /* The backward function takes the pointer of gradients(*grad_input) from NEXT layer
    and perform the gradient calculation. The calculated gradients will be saved
    in the gradients data member. It also calculates the gradients that flows to
    the previous layer and saves in data member and saves the pointer in *grad_output*/
    virtual void backward(float* grad_input, float* grad_output) = 0;

    // INPUT SIZE
    /*Returns the Input size the layer*/
    virtual size_t getInputSize() = 0;

    // OUTPUT SIZE
    /*Returns the Output size the layer*/
    virtual size_t getOutputSize() = 0;

    // NUMBER OF WEIGHTS SIZE
    /*Returns the number of weights or bias in the layer*/
    virtual size_t numParams() = 0;

    // LAYER NAME
    /*Returns the name of the layer (Linear/Softmax/RelU)*/
    virtual string getLayerName() = 0;

    // SET DEVICE
    /* This function will set the device flag of the layer and move all the 
    memory allocated (weights, gradients, buffers to the specified device and link
    it to specific device pointers)*/
    virtual void setDevice(int device) = 0;

    // GET DEVICE
    /* Returns the device state of the layer*/
    virtual int getDevice() = 0;
    

    // -------------------------------------------------------------------
    // OPTIONAL FUNCTIONS
    // -------------------------------------------------------------------

    // UPDATE THE PARAMETERS
    /* Carries out hte Gradient Update operation using the weights and gradients
    calculated. It updates the weights and biases with the updated weights and biases*/
    virtual void updateParameters(float learning_rate){}

    // Initialize the weights and Biases
    /* This function will initialize the weights and biases of the layer based on
    input and output size. It will be used in the constructor of the layer and allocate
    memory using the host pointer.*/
    /* This function will be used in the constructor of the layer and allocate*/
    virtual void initializeWeights(){}; 
    virtual void initializeBiases(){}; 

};


// CUDA KERNEL IMPLEMENTATION OUTSIDE THE CLASS
// __global__ void forwardKernel(){}; // Example of a CUDA kernel function
// __global__ void backwardKernel(){}; // Example of a CUDA kernel function

// CPU IMPLEMENTATION 
// void forwardCPU(){}; // Example of a CPU function
// void backwardCPU(){}; // Example of a CPU function


#endif // BASE_LAYER_H