# include "../../include/cnn_library/layers/linear.h"
# include <iostream>

// Constructor
Linear::Linear(size_t input_size, size_t output_size, size_t batch_size) {
    this->layer_name = "Linear";
    this->input_size = input_size;
    this->output_size = output_size;
    this->batch_size = batch_size;

    // Allocate memory for weights and biases
    size_t weights_size = input_size * output_size;
    host_weights = new float[weights_size];
    host_biases = new float[output_size];

    // Allocate memory for gradients
    host_grad_weights = new float[weights_size];
    host_grad_biases = new float[output_size];

    // Allocate buffers for forward and backward passes
    host_forward_buffer = new float[output_size * batch_size];
    host_backward_buffer = new float[input_size * batch_size];

    // Initialize device pointers to nullptr
    device_weights = nullptr;
    device_biases = nullptr;
    device_grad_weights = nullptr;
    device_grad_biases = nullptr;
    device_forward_buffer = nullptr;
    device_backward_buffer = nullptr;

    // Allocate cached input
    cached_input = new float[input_size * batch_size];

    // Initialize weights and biases
    initializeWeights();
    initializeBiases();

    // TODO: Is this necessary?
    // Initialize gradients to zero
    std::memset(host_grad_weights, 0, weights_size * sizeof(float));
    std::memset(host_grad_biases, 0, output_size * sizeof(float));

    std::cout << "Linear layer created with input size: " << input_size 
              << ", output size: " << output_size 
              << ", batch size: " << batch_size << std::endl;
}

// Destructor
Linear::~Linear() {
    // Free host memory
    delete[] host_weights;
    delete[] host_biases;
    delete[] host_grad_weights;
    delete[] host_grad_biases;
    delete[] host_forward_buffer;
    delete[] host_backward_buffer;
    delete[] cached_input;

    if (device) {
        // TODO: Free CUDA memory
    }

    std::cout << "Linear layer destroyed" << std::endl;
}

// Forward
void Linear::forward(float* input, float* output) {
    // Cache input for backward pass
    std::memcpy(cached_input, input, input_size * batch_size * sizeof(float));

    if(!device) {
        // CPU forward pass
        forwardCPU(input, host_forward_buffer);
    }
    else {
        // TODO: GPU forward pass

    }
}


// CPU forward pass
void Linear::forwardCPU(float* input, float* output) {
    // Initialize output with biases
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < output_size; o++) {
            output[b * output_size + o] = host_biases[o];
        }
    }

    // Matrix multiplication: output += input * weights
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < input_size; i++) {
            for (size_t o = 0; o < output_size; o++) {
                output[b * output_size + o] += input[b * input_size + i] * host_weights[i * output_size + o];
            }
        }
    }
}

// Backward
void Linear::backward(float* grad_input, float* grad_output) {
    if(!device) {
        // CPU implementation
        backwardCPU(grad_input, grad_output);
    } else {
        // TODO: GPU backward pass
    }
}

// CPU backward pass
void Linear::backwardCPU(float* grad_input, float* grad_output) {
    // Initialize grad_output with zeros
    std::memset(grad_output, 0, input_size * batch_size * sizeof(float));

    // Compute di
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < input_size; i++) {
            for (size_t o = 0; o < output_size; o++) {
                grad_output[b * input_size + i] += grad_input[b * output_size + o] * host_weights[i * output_size + o];
            }
        }
    }

    // Compute dw
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < input_size; i++) {
            for (size_t o = 0; o < output_size; o++) {
                host_grad_weights[i * output_size + o] += 
                    cached_input[b * input_size + i] * grad_input[b * output_size + o];
            }
        }
    }

    // Compute db
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < output_size; o++) {
            host_grad_biases[o] += grad_input[b * output_size + o];
        }
    }
}

// Set device (0 - CPU, 1 - GPU)
// TODO: Change `device` to enum
void Linear::setDevice(int device) {
    this->device = device;

    if (device) {
        // TODO: Allocate and copy memory to GPU
    }
}

// Update weights with gradients
void Linear::updateParameters(float learning_rate) {
    if (!device) {
        // Update weights on CPU
        for (size_t i = 0; i < input_size * output_size; i++) {
            host_weights[i] -= learning_rate * host_grad_weights[i];
        }
        
        // Update biases on CPU
        for (size_t i = 0; i < output_size; i++) {
            host_biases[i] -= learning_rate * host_grad_biases[i];
        }

        // Reset gradients
        std::memset(host_grad_weights, 0, input_size * output_size * sizeof(float));
        std::memset(host_grad_biases, 0, output_size * sizeof(float));
    } else {
        // TODO: GPU impl
    }
}

// Initialize weights with Xavier initialization
void Linear::initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Xavier init: stddev = sqrt(2 / (input_size + output_size))
    float stddev = std::sqrt(2.0f / (input_size | output_size));
    std::normal_distribution<float> d(0.0f, stddev);

    // Initialize weights
    for (size_t i = 0; i < input_size * output_size; i++) {
        host_weights[i] = d(gen);
    }
}

// Initialize biases to zero
void Linear::initializeBiases() {
    std::memset(host_biases, 0, output_size * sizeof(float));
}

// Get input size
size_t Linear::getInputSize() {
    return input_size;
}

// Get output size
size_t Linear::getOutputSize() {
    return output_size;
}

// Get number of parameters (weights + biases)
size_t Linear::numParams() {
    return (input_size * output_size) + output_size;
}

// Get layer name
std::string Linear::getLayerName() {
    return layer_name;
}

// Get device
int Linear::getDevice() {
    return device;
}

void Linear::forwardGPU(float* input, float* output) {

}

void Linear::backwardGPU(float* grad_input, float* grad_output) {

}
