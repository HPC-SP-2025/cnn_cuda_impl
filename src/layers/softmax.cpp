#include "../../include/cnn_library/layers/softmax.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// Constructor
Softmax::Softmax(size_t num_classes, size_t batch_size) {
    this->layer_name = "Softmax";
    this->input_size = num_classes;
    this->output_size = num_classes;
    this->batch_size = batch_size;

    this->host_forward_buffer = (float *)malloc(sizeof(float) * this->output_size * this->batch_size);
    this->host_backward_buffer = (float *)malloc(sizeof(float) * this->input_size * this->batch_size);
    this->device_forward_buffer = nullptr;
    this->device_backward_buffer = nullptr;

    this->test_mode = 0; // 0 for false, 1 for true i.e. test mode active

    std::cout << "Softmax constructor call\n";
}

// Destructor
Softmax::~Softmax() {
    free(host_forward_buffer);
    free(host_backward_buffer);

    std::cout << "Softmax destructor call\n";
}

// Forward
float *Softmax::forward(float *input) {
    if (!device) {
        forwardCpuSoftmax(input, this->host_forward_buffer);
        return this->host_forward_buffer;
    }
}

// Backward
float *Softmax::backward(float *grad_input) {
    if (!device) {
        backwardCpuSoftmax(grad_input, this->host_backward_buffer);
        return this->host_backward_buffer;
    }
}

size_t Softmax::getInputSize() { return this->input_size; }
size_t Softmax::getOutputSize() { return this->output_size; }
size_t Softmax::numParams() { return 0; }
string Softmax::getLayerName() { return this->layer_name; }

// Set device
void Softmax::setDevice(int device) { this->device = device; }

int Softmax::getDevice() { return this->device; }

void Softmax::setTestMode(int test_mode) {
    this->test_mode = test_mode; // 0 for false, 1 for true i.e. test mode active
}

void Softmax::forwardCpuSoftmax(float *input, float *output) {
#pragma omp parallel for
    for (size_t b = 0; b < this->batch_size; ++b) {
        float *current_input = input + b * this->input_size;
        float *current_output = output + b * this->output_size;

        // Find max for numerical stability
        float max_val = *std::max_element(current_input, current_input + this->input_size);

        float sum = 0.0f;
        for (size_t i = 0; i < this->output_size; ++i) {
            current_output[i] = exp(current_input[i] - max_val);
            sum += current_output[i];
        }

        // Just to be sure, can only happen if all val in exp(val) are -inf
        if (sum == 0.0f)
            sum = 1e-6f; // TODO: Maybe define this constant globally?

        // Normalize
        for (size_t i = 0; i < this->output_size; ++i) {
            current_output[i] /= sum;
        }
    }
}

void Softmax::backwardCpuSoftmax(float *grad_input, float *grad_output) {

#pragma omp parallel for
    for (size_t i = 0; i < this->output_size * batch_size; ++i) {
        grad_output[i] = grad_input[i];
    }
}
