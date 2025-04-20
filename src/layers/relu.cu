#include "../../include/cnn_library/layers/relu.h"
#include <iostream>
#include <string>

// Layer name
string ReLU::getLayerName(){
    return this->layer_name;
}

// Constructor
// Should allocate CPU output memory
ReLU::ReLU(size_t input_size, size_t output_size, size_t batch_size){   
    this->layer_name = "ReLU";
    this->input_size = input_size;
    this->output_size = output_size;
    this-> batch_size = batch_size;
    
    this->host_forward_buffer = (float*)malloc(sizeof(float) * output_size * batch_size);
    this->host_backward_buffer = (float*)malloc(sizeof(float) * input_size * batch_size);
    std::cout << "ReLU constructor call\n";
}

// Destructor
// Should delete CPU (& GPU, if exists) output memory
ReLU::~ReLU(){  
    free(host_forward_buffer);
    free(host_backward_buffer);
    if(device){ 
        cudaFree(device_forward_buffer);
        cudaFree(device_backward_buffer); 
    }
    std::cout << "ReLU destructor call\n";
}

// CUDA kernel declaration
__global__ void forwardKernelReLU(float* input, float* output, size_t output_size, size_t batch_size);
__global__ void backwardKernelReLU(float* grad_input, float* grad_output, float* layer_input_ptr, size_t input_size, size_t batch_size);

// Forward
float* ReLU::forward(float* input){
    if(!device){
        forwardCpuReLU(input, host_forward_buffer);
        return host_forward_buffer;
    }
    else{
        size_t blocks = (output_size + threads_per_block - 1) / threads_per_block;
        forwardKernelReLU<<< blocks, threads_per_block >>>(input, device_forward_buffer, output_size, batch_size);
        return device_forward_buffer;
    }
    layer_input_ptr = input;
}

// Backward
void ReLU::backward(float* grad_input, float* grad_output){
    if(!device){
        backwardCpuReLU(grad_input, host_backward_buffer);
        grad_output = host_backward_buffer;
    }
    else{
        size_t blocks = (input_size + threads_per_block - 1) / threads_per_block;
        backwardKernelReLU<<< blocks, threads_per_block >>>(grad_input, device_backward_buffer, layer_input_ptr, input_size, batch_size);
        grad_output = device_backward_buffer;
    }
}

// Set device
// Should allocate CUDA memory only if device is GPU
void ReLU::setDevice(int device){   
    this->device = device;

    if(device){
        cudaMalloc(&device_forward_buffer, sizeof(float) * output_size * batch_size);
        cudaMalloc(&device_backward_buffer, sizeof(float) * input_size * batch_size);
        cudaMemcpy(device_forward_buffer, host_forward_buffer, sizeof(float)*output_size*batch_size, cudaMemcpyHostToDevice);
        cudaMemcpy(device_backward_buffer, host_backward_buffer, sizeof(float)*input_size*batch_size, cudaMemcpyHostToDevice);
    }
}

// Get device
int ReLU::getDevice(){
    return this->device;
}

// Getter functions
size_t ReLU::getInputSize() { return input_size; }
size_t ReLU::getOutputSize() { return output_size; }

// CPU forward implementation
void ReLU::forwardCpuReLU(float* input, float* output){
    for (size_t i=0; i<output_size; i++){
        output[i] = fmaxf(input[i],0.0f);
    }
}

// CPU backward implementation
void ReLU::backwardCpuReLU(float* grad_input, float* grad_output){
    float grad;
    for (size_t i=0; i<input_size; i++){
        grad = (layer_input_ptr[i] > 0) ? 1.0f : 0.0f;
        grad_output[i] = grad_input[i] * grad;
    }
}

// GPU forward implementation
__global__ void forwardKernelReLU(float* input, float* output, size_t output_size, size_t batch_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size*batch_size) {
        output[idx] = fmaxf(input[idx],0.0f);
    }
} 

// GPU backward implementation
__global__ void backwardKernelReLU(float* grad_input, float* grad_output, float* layer_input_ptr, size_t input_size, size_t batch_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size*batch_size) {
        float grad = (layer_input_ptr[idx] > 0) ? 1.0f : 0.0f;
        grad_output[idx] = grad_input[idx] * grad;
    }
} 