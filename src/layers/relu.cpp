#include "../../include/cnn_library/layers/relu.h"
#include <cmath>
#include <iostream>
#include <string>

// Constructor
ReLU::ReLU(size_t input_size, size_t output_size, size_t batch_size) {
    this->layer_name = "ReLU";
    this->input_size = input_size;
    this->output_size = output_size;
    this->batch_size = batch_size;
    this->threads_per_block = 32;

    this->host_forward_buffer = (float *)malloc(sizeof(float) * this->output_size * this->batch_size);
    this->host_backward_buffer = (float *)malloc(sizeof(float) * this->input_size * this->batch_size);
    this->device_forward_buffer = nullptr;
    this->device_backward_buffer = nullptr;
    std::cout << "ReLU constructor call\n";
}

// Destructor
ReLU::~ReLU() {
    free(host_forward_buffer);
    free(host_backward_buffer);

    std::cout << "ReLU destructor call\n";
}

// Forward
float *ReLU::forward(float *input) {
    this->layer_input_ptr = input;
    if (!device) {
        forwardCpuReLU(input, this->host_forward_buffer);
        return this->host_forward_buffer;
    }
}

// Backward
float *ReLU::backward(float *grad_input) {
    if (!device) {
        backwardCpuReLU(grad_input, this->host_backward_buffer);
        return this->host_backward_buffer;
    }
}

// Getter functions
size_t ReLU::getInputSize() { return this->input_size; }
size_t ReLU::getOutputSize() { return this->output_size; }
size_t ReLU::numParams() { return 0; }
string ReLU::getLayerName() { return this->layer_name; }
int ReLU::getDevice() { return this->device; }

// Set device
void ReLU::setDevice(int device) { this->device = device; }

// CPU forward implementation
void ReLU::forwardCpuReLU(float *input, float *output) {

#pragma omp parallel for
    for (size_t i = 0; i < output_size * batch_size; i++) {
        output[i] = fmaxf(input[i], 0.0f);
    }
}

// CPU backward implementation
void ReLU::backwardCpuReLU(float *grad_input, float *grad_output) {

#pragma omp parallel for
    for (size_t i = 0; i < input_size * batch_size; i++) {
        grad_output[i] = (this->layer_input_ptr[i] > 0) ? grad_input[i] : 0.0f;
    }
}
