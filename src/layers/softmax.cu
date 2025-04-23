#include "../../include/cnn_library/layers/softmax.h"
#include <iostream>
#include <cmath>
#include <algorithm>



// Constructor
Softmax::Softmax(size_t num_classes, size_t batch_size) {
    this->layer_name = "Softmax";
    this->input_size = num_classes;
    this->output_size = num_classes;
    this->batch_size = batch_size;

    this->host_forward_buffer = (float*)malloc(sizeof(float) * this->output_size * this->batch_size);
    this->host_backward_buffer = (float*)malloc(sizeof(float) * this->input_size * this->batch_size);
    this->device_forward_buffer = nullptr;
    this->device_backward_buffer = nullptr;
    std::cout << "Softmax constructor call\n";
}

// Destructor
Softmax::~Softmax() {
    free(host_forward_buffer);
    free(host_backward_buffer);
    if(device){
        cudaFree(device_forward_buffer);
        cudaFree(device_backward_buffer);
    }
    std::cout << "Softmax destructor call\n";
}

// Forward
float* Softmax::forward(float* input) {
    if (!device) {
        forwardCpuSoftmax(input, this->host_forward_buffer);
        return this->host_forward_buffer;
    }
    else {
        forwardGpuSoftmax(input, this->device_forward_buffer);
        return this->device_forward_buffer;
    }
}

// Backward
float* Softmax::backward(float* grad_input) {
    if (!device) {
        backwardCpuSoftmax(grad_input, this->host_backward_buffer);
        return this->host_backward_buffer;
    }
    else {
        backwardGpuSoftmax(grad_input, this->device_backward_buffer);
        return this->device_backward_buffer;
    }
}

size_t Softmax::getInputSize() { return this->input_size; }
size_t Softmax::getOutputSize() { return this->output_size; }
size_t Softmax::numParams() { return 0; }
string Softmax::getLayerName() { return this->layer_name; }

// Set device
void Softmax::setDevice(int device) {
    this->device = device;

    if (device) {
        cudaMalloc((void**)&this->device_forward_buffer, sizeof(float) * this->output_size * this->batch_size);
        cudaMalloc((void**)&this->device_backward_buffer, sizeof(float) * this->input_size * this->batch_size);
        cudaMemcpy(this->device_forward_buffer, this->host_forward_buffer, sizeof(float) * this->output_size * this->batch_size, cudaMemcpyHostToDevice);
        cudaMemcpy(this->device_backward_buffer, this->host_backward_buffer, sizeof(float) * this->input_size * this->batch_size, cudaMemcpyHostToDevice);
    }
}

int Softmax::getDevice() { return this->device; }

void Softmax::forwardCpuSoftmax(float* input, float* output) {
    for (size_t b = 0; b < this->batch_size; ++b) {
        float* current_input = input + b * this->input_size;
        float* current_output = output + b * this->output_size;

        // Find max for numerical stability
        float max_val = *std::max_element(current_input, current_input + this->input_size);

        float sum = 0.0f;
        for (size_t i = 0; i < this->output_size; ++i) {
            current_output[i] = exp(current_input[i] - max_val);
            sum += current_output[i];
        }

        // Just to be sure, can only happen if all val in exp(val) are -inf
        if (sum == 0.0f) sum = 1e-6f; // TODO: Maybe define this constant globally?

        // Normalize
        for (size_t i = 0; i < this->output_size; ++i) {
            current_output[i] /= sum;
        }
    }
}

void Softmax::backwardCpuSoftmax(float* grad_input, float* grad_output) {
    for (size_t i=0; i < this->output_size * batch_size; ++i) {
        grad_output[i] = grad_input[i];
    }
}


void Softmax::forwardGpuSoftmax(float* input, float* output) {
    size_t blocks = (this->batch_size + 1024 - 1) / 1024;
    size_t threads_per_block = (this->batch_size + blocks - 1) / blocks;
	forwardKernelSoftmax<<<blocks, threads_per_block>>>(input, output, this->output_size, this->batch_size);
	cudaDeviceSynchronize();
}

void Softmax::backwardGpuSoftmax(float* grad_input, float* grad_output) {
    size_t blocks = (this->batch_size + 1024 - 1) / 1024;
    size_t threads_per_block = (this->batch_size + blocks - 1) / blocks;
	forwardKernelSoftmax<<<blocks, threads_per_block>>>(grad_input, grad_output, this->input_size, this->batch_size);
	cudaDeviceSynchronize();
}

__global__ void forwardKernelSoftmax(float* input, float* output, size_t num_classes, size_t batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        size_t offset = idx * num_classes;
        float* current_input = input + offset;
        float* current_output = output + offset;

        // Find max for numerical stability
        float max_val = *std::max_element(current_input, current_input + num_classes);

        float sum = 0.0f;
        for (size_t i = 0; i < num_classes; ++i) {
            current_output[i] = exp(current_input[i] - max_val);
            sum += current_output[i];
        }

        // Just to be sure, can only happen if all val in exp(val) are -inf
        if (sum == 0.0f) sum = 1e-6f; // TODO: Maybe define this constant globally?

        // Normalize
        for (size_t i = 0; i < num_classes; ++i) {
            current_output[i] /= sum;
        }
    }
}

__global__ void backwardKernelSoftmax(float* grad_input, float* grad_output, size_t num_classes, size_t batch_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        size_t offset = idx * num_classes;
        float* current_grad_input = grad_input + offset;
        float* current_grad_output = grad_output + offset;
        for (size_t i = 0; i < num_classes; ++i) {
        	current_grad_output[i] = current_grad_input[i];
        }
    }
}
