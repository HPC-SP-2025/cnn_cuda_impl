#include "../../include/cnn_library/layers/linear.h"
#include <cstring>>
#include <iostream>

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
    // cached_input = new float[input_size * batch_size];

    // Initialize weights and biases
    initializeWeights();
    initializeBiases();

    // Initialize gradients to zero
    std::memset(host_grad_weights, 0, weights_size * sizeof(float));
    std::memset(host_grad_biases, 0, output_size * sizeof(float));

    std::cout << "Linear layer created with input size: " << input_size << ", output size: " << output_size
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
    // delete[] cached_input;

    if (device) {
        // TODO: Free CUDA memory
    }

    std::cout << "Linear layer destroyed" << std::endl;
}

// Forward
float *Linear::forward(float *input) {
    // Cache input for backward pass
    // std::memcpy(cached_input, input, input_size * batch_size * sizeof(float));

    cached_input = input;

    if (!device) {
        // CPU forward pass
        return forwardCPU(input);
    } else {
        // TODO: GPU forward pass
        return forwardGPU(input);
    }
}

// CPU forward pass
float *Linear::forwardCPU(float *input) {
    // Initialize output with biases
    float *output = host_forward_buffer;
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

    return output;
}

// Backward
float *Linear::backward(float *grad_input) {
    if (!device) {
        // CPU implementation
        return backwardCPU(grad_input);
    } else {
        // TODO: GPU backward pass
        return backwardGPU(grad_input);
    }
}

// CPU backward pass
float *Linear::backwardCPU(float *grad_input) {
    // Initialize grad_output with zeros
    std::memset(host_backward_buffer, 0, input_size * batch_size * sizeof(float));

    // Compute di
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < input_size; i++) {
            for (size_t o = 0; o < output_size; o++) {
                host_backward_buffer[b * input_size + i] +=
                    grad_input[b * output_size + o] * host_weights[i * output_size + o];
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

    return host_backward_buffer;
}

// Set device (0 - CPU, 1 - GPU)
// TODO: Change `device` to enum
void Linear::setDevice(int device) {
    this->device = device;

    if (device) {
        // TODO: Allocate and copy memory to GPU
        size_t weights_size = input_size * output_size;
        size_t input_bytes = input_size * batch_size * sizeof(float);
        size_t output_bytes = output_size * batch_size * sizeof(float);

        cudaMalloc(&device_weights, weights_size * sizeof(float));
        cudaMalloc(&device_biases, output_size * sizeof(float));
        cudaMalloc(&device_grad_weights, weights_size * sizeof(float));
        cudaMalloc(&device_grad_biases, output_size * sizeof(float));
        cudaMalloc(&device_forward_buffer, output_bytes);
        cudaMalloc(&device_backward_buffer, input_bytes);

        cudaMemcpy(device_weights, host_weights, weights_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_biases, host_biases, output_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(device_grad_weights, 0, weights_size * sizeof(float));
        cudaMemset(device_grad_biases, 0, output_size * sizeof(float));
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
        int weight_size = input_size * output_size;
        int bias_size = output_size;
        int threads = 256;
        int blocks = (input_size * output_size + threads - 1) / threads;

        update_parameters_kernel<<<blocks, threads>>>(device_weights, device_grad_weights, device_biases,
                                                      device_grad_biases, learning_rate, weight_size, bias_size);
    }
}

// Initialize weights with Xavier initialization
void Linear::initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Xavier init: stddev = sqrt(2 / (input_size + output_size))
    float stddev = std::sqrt(2.0f / (input_size + output_size));
    std::normal_distribution<float> d(0.0f, stddev);

    // Initialize weights
    for (size_t i = 0; i < input_size * output_size; i++) {
        host_weights[i] = d(gen);
    }
}

// Initialize biases to zero
void Linear::initializeBiases() { std::memset(host_biases, 0, output_size * sizeof(float)); }

// Get input size
size_t Linear::getInputSize() { return input_size; }

// Get output size
size_t Linear::getOutputSize() { return output_size; }

// Get number of parameters (weights + biases)
size_t Linear::numParams() { return (input_size * output_size) + output_size; }

// Get layer name
std::string Linear::getLayerName() { return layer_name; }

// Get device
int Linear::getDevice() { return device; }

// Set weights from an external array
void Linear::setWeights(float *weights) {
    std::memcpy(host_weights, weights, input_size * output_size * sizeof(float));

    if (device) {
        cudaMemcpy(device_weights, weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    }
}

// Set biases from an external array
void Linear::setBiases(float *biases) {
    std::memcpy(host_biases, biases, output_size * sizeof(float));

    if (device) {
        cudaMemcpy(device_biases, biases, output_size * sizeof(float), cudaMemcpyHostToDevice);
    }
}

// Get weights by copying to an external array
void Linear::getWeights(float *weights) {
    if (device) {
        cudaMemcpy(host_weights, device_weights, input_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    std::memcpy(weights, host_weights, input_size * output_size * sizeof(float));
}

// Get biases by copying to an external array
void Linear::getBiases(float *biases) {
    if (device) {
        cudaMemcpy(host_biases, device_biases, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    std::memcpy(biases, host_biases, output_size * sizeof(float));
}

float *Linear::forwardGPU(float *input) {
    size_t input_bytes = input_size * batch_size * sizeof(float);
    cached_input = input;

    dim3 blockDim(16, 16);
    dim3 gridDim((output_size + 15) / 16, (batch_size + 15) / 16);

    forward_kernel<<<gridDim, blockDim>>>(input, device_weights, device_biases, device_forward_buffer, input_size,
                                          output_size, batch_size);

    return device_forward_buffer;
}

float *Linear::backwardGPU(float *grad_input) {
    size_t input_bytes = input_size * batch_size * sizeof(float);
    size_t output_bytes = output_size * batch_size * sizeof(float);

    dim3 blockDim(16, 16);

    // grad_input
    dim3 grid_input((input_size + 15) / 16, (batch_size + 15) / 16);
    backward_input_kernel<<<grid_input, blockDim>>>(grad_input, device_weights, device_backward_buffer, input_size,
                                                    output_size, batch_size);

    // grad_weights
    dim3 grid_w((output_size + 15) / 16, (input_size + 15) / 16);
    backward_weight_kernel<<<grid_w, blockDim>>>(cached_input, grad_input, device_grad_weights, input_size, output_size,
                                                 batch_size);

    // grad_biases
    dim3 grid_b((output_size + 15) / 16);
    backward_bias_kernel<<<grid_b, 256>>>(grad_input, device_grad_biases, output_size, batch_size);

    return device_backward_buffer;
}

__global__ void forward_kernel(float *input, float *weights, float *biases, float *output, size_t input_size,
                               size_t output_size, size_t batch_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // output index

    if (row < batch_size && col < output_size) {
        float val = biases[col];
        for (int i = 0; i < input_size; ++i) {
            val += input[row * input_size + i] * weights[i * output_size + col];
        }
        output[row * output_size + col] = val;
    }
}

__global__ void backward_input_kernel(float *grad_output, float *weights, float *grad_input, size_t input_size,
                                      size_t output_size, size_t batch_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // input index

    if (row < batch_size && col < input_size) {
        float val = 0.0f;
        for (int j = 0; j < output_size; ++j) {
            val += grad_output[row * output_size + j] * weights[col * output_size + j];
        }
        grad_input[row * input_size + col] = val;
    }
}

__global__ void backward_weight_kernel(float *input, float *grad_output, float *grad_weights, size_t input_size,
                                       size_t output_size, size_t batch_size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // input index
    int o = blockIdx.x * blockDim.x + threadIdx.x; // output index

    if (i < input_size && o < output_size) {
        float val = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            val += input[b * input_size + i] * grad_output[b * output_size + o];
        }
        atomicAdd(&grad_weights[i * output_size + o], val);
    }
}

__global__ void backward_bias_kernel(float *grad_output, float *grad_biases, size_t output_size, size_t batch_size) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o < output_size) {
        float val = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            val += grad_output[b * output_size + o];
        }
        atomicAdd(&grad_biases[o], val);
    }
}
__global__ void update_parameters_kernel(float *weights, float *grad_weights, float *biases, float *grad_biases,
                                         float learning_rate, size_t weight_size, size_t bias_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Update weights
    if (idx < weight_size) {
        weights[idx] -= learning_rate * grad_weights[idx];
        grad_weights[idx] = 0.0f;
    }

    // Update biases
    if (idx < bias_size) {
        biases[idx] -= learning_rate * grad_biases[idx];
        grad_biases[idx] = 0.0f;
    }
}
