#ifndef BASE_LAYER_H
#define BASE_LAYER_H

// Include the headers
#include <memory>
#include <string>
#include <vector>

using namespace std;

class Layer {

  protected:
    string layer_name; // Name of the layer
    int device = 0;    // 0 for CPU, 1 for GPU
    size_t input_size;
    size_t output_size;
    size_t batch_size;
    unsigned int threads_per_block = 1024;

    // For convolution Layer only
    unsigned int input_height;
    unsigned int output_height;
    unsigned int input_width;
    unsigned int output_width;

    // Weights and Biases
    float *host_weights;
    float *host_biases;
    float *device_weights;
    float *device_biases;

    // Gradients
    float *host_grad_weights;
    float *host_grad_biases;
    float *device_grad_weights;
    float *device_grad_biases;

    // Forward Buffer
    float *host_forward_buffer;
    float *device_forward_buffer;

    // Backward Buffer
    float *host_backward_buffer;
    float *device_backward_buffer;

  public:
    // Constructor:
    /* Constructor is responsible for inilializing the weights and biases, gradients,
    forward_buffer and the backward_buffer for the layer on the CPU*/
    Layer() {};
    virtual ~Layer() = default;

    // -------------------------------------------------------------------
    // COMPULSORY FUNCTIONS
    // -------------------------------------------------------------------

    // FORWARD FUNCTION
    /* Perfroms the forward pass of the layer and saves output in its output
    data member.

    Arguements:
    *input: pointer of the output of the previous layer
    *output: pointer of the output of the current layer
    */

    virtual float *forward(float *input) = 0;

    // BACKWARD FUNCTION
    /* Performs the backward pass of the layer and saves the gradients in its
    gradients data member. It also calculates the gradients that flows to
    the PREVIOUS layer and saves in data member and passes the pointer in *grad_output
    Arguements:
    *grad_input: pointer of the gradients from the NEXT layer
    *grad_output: pointer of the gradients to be passed to the PREVIOUS layer

    */
    virtual float *backward(float *grad_input) = 0;

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
    /* This function will set the device flag of the layer and copy all the
    memory allocated (weights, gradients, buffers to the specified device and link
    it to specific device pointers)*/
    virtual void setDevice(int device) = 0;

    // GET DEVICE
    /* Returns the device state of the layer*/
    virtual int getDevice() = 0;

    // SET WEIGHTS
    /* This function will set the weights of the layer from a loaded file */
    virtual void setParameters(const vector<float>& parameters) = 0;

    // -------------------------------------------------------------------
    // OPTIONAL FUNCTIONS
    // -------------------------------------------------------------------

    // UPDATE THE PARAMETERS
    /* Carries out the Gradient Update operation using the weights and gradients
    calculated. It updates the weights and biases with the updated weights and biases*/
    virtual void updateParameters(float learning_rate) {}

    // Initialize the weights and Biases
    /* This function will initialize the weights and biases of the layer based on
    input and output size. It will be used in the constructor of the layer and allocate
    memory using the host pointer.*/
    /* This function will be used in the constructor of the layer and allocate*/
    virtual void initializeWeights() {};
    virtual void initializeBiases() {};
};

// CUDA KERNEL IMPLEMENTATION OUTSIDE THE CLASS
// __global__ void forwardKernel(){}; // Example of a CUDA kernel function
// __global__ void backwardKernel(){}; // Example of a CUDA kernel function

// CPU IMPLEMENTATION
// void forwardCPU(){}; // Example of a CPU function
// void backwardCPU(){}; // Example of a CPU function

#endif // BASE_LAYER_H