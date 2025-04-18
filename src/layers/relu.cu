#include "../../include/cnn_library/layers/relu.h"
#include <iostream>

// Constructor
ReLU::ReLU(int input_size, int output_size){   // Should allocate CPU input & output memory
    this->input_size = input_size;
    this->output_size = output_size;

    float* host_output_buffer = (float*)malloc(sizeof(float) * output_size);
    cout << "ReLU constructor call\n";
}

// Destructor
ReLU::~ReLU(){  // Should delete CPU & GPU (if exists) input & output memory
    free(host_output_buffer);
    if(device){ cudeFree(device_output_buffer); }

    cout << "ReLU destructor call\n";
}

void ReLU::forward(float* input, float* output){
    if(!device){
        forwardCpuReLU(input, host_output_buffer);
        output = host_output_buffer;
    }
    else{
        forwardKernelReLU<<< blocks, threads_per_block >>>(input, device_output_buffer);
        output = device_output_buffer;
    }
}

void ReLU::backward(float* grad_input, float* grad_output){
    if(!device){
        backwardCpuReLU();
    }
    else{
        backwardKernelReLU<<< blocks, threads_per_block >>>();
    }
}

void ReLU::setDevice(int device){   // Should allocate CUDA memory only if device is GPU
    this->device = device;

    if(device){
        cudaMalloc(&device_output_buffer, sizeof(float) * output_size);
        cudaMemcpy(device_output_buffer, host_output_buffer, sizeof(float)*output_size, cudaMemcpyHostToDevice);
    }
}

// Getter functions
size_t ReLU::getInputSize() { return input_size; }
size_t ReLU::getOutputSize() { return output_size; }

// CPU forward implementation
void ReLU::forwardCpuReLU(float* input, float* output){
    for (int i=0; i<input_size; i++){
        output[i] = fmaxf(input[i],0.0f);
    }
}

// CPU backward implementation
void ReLU::backwardCpuReLU(){

}

// GPU forward implementation
void ReLU::forwardKernelReLU(float* input, float* output){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = fmaxf(input[idx],0.0f);
} 

// GPU backward implementation
void ReLU::backwardKernelReLU(){

} 