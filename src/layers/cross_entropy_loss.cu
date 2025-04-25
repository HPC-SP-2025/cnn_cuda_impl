#include "../../include/cnn_library/layers/cross_entropy_loss.h"
#include <iostream>
using namespace std;

#define BLOCK_SIZE 256
#define EPSILON 1e-9

// TODO: use reduction
__global__ void forward_cel_kernel(const float *pred, float *target, float *loss, int n, int num_classes) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    float loss_i = 0.0;
    if (idx < n) {
        loss_i = -log(max(pred[idx * num_classes + (int)target[idx]], EPSILON));
        // atomicAdd(loss, -loss_i);
    }
    extern __shared__ float shared_loss[];
    shared_loss[tid] = loss_i;

    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            shared_loss[tid] += shared_loss[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(loss, shared_loss[0]);
}

/**
 * output: Flattened 1D vector with class probabilites
 * target: class label 0-indexed
 */
float Cross_Entropy_Loss::forward_CPU(const float *pred, float *target) {
    int n = batch_size;
    int num_classes = input_size;

    float loss = 0.0;
    for (auto i = 0; i < n; i++) {
        loss -= log(max(pred[i * num_classes + (int)target[i]], EPSILON));
    }
    loss /= static_cast<float>(n);
    return loss;
}

float Cross_Entropy_Loss::forward_GPU(const float *pred, float *target) {
    float loss = 0;
    cudaMemset(d_loss, 0, sizeof(float));
    int num_blocks = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int mem_size = threads_per_block * sizeof(float);
    forward_cel_kernel<<<num_blocks, BLOCK_SIZE, mem_size>>>(pred, this->target, d_loss, batch_size, input_size);
    cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    return loss;
}

__global__ void backward_cel_kernel(float *grad_output, float *pred, float *target, int n, int num_classes) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        for (auto j = 0; j < num_classes; j++)
            grad_output[idx * num_classes + j] = pred[idx * num_classes + j] / n;

        grad_output[idx * num_classes + (int)target[idx]] -= 1.0 / n;
    }
}

void Cross_Entropy_Loss::backward_CPU(float *grad_output, float *pred, float *target) {

    int n = batch_size;
    int num_classes = input_size;
    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < num_classes; j++)
            grad_output[i * num_classes + j] = pred[i * num_classes + j] / n;

        grad_output[i * num_classes + (int)target[i]] -= 1.0 / n;
    }
}

Cross_Entropy_Loss::Cross_Entropy_Loss(size_t num_classes, size_t batch_size) {
    this->layer_name = "Cross_Entropy_Loss";
    this->batch_size = batch_size;
    this->input_size = num_classes;

    this->host_forward_buffer = new float[1];
    this->host_backward_buffer = new float[batch_size * num_classes];
}

Cross_Entropy_Loss::~Cross_Entropy_Loss() {
    delete[] host_forward_buffer;
    delete[] host_backward_buffer;
    if (device) {
        cudaFree(device_backward_buffer);
        cudaFree(this->d_loss);
    }
}

float *Cross_Entropy_Loss::forward(float *pred) {

    float loss;
    if (this->device) {
        
        loss = forward_GPU(pred, this->target);
    } 
    else {
        loss = forward_CPU(pred, this->target);
    }
    host_forward_buffer[0] = loss;
    return host_forward_buffer;
}

float *Cross_Entropy_Loss::backward(float *pred) {
    if (this->device) {
        int num_blocks = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        backward_cel_kernel<<<num_blocks, BLOCK_SIZE>>>(device_backward_buffer, pred, this->target, batch_size,
                                                        input_size);
        return device_backward_buffer;
    } else {
        backward_CPU(host_backward_buffer, pred, this->target);
        return host_backward_buffer;
    }
}

void Cross_Entropy_Loss::setDevice(int device) {
    this->device = device;
    if (device) {
        cudaMalloc(&d_loss, sizeof(float));
        cudaMalloc(&device_backward_buffer, sizeof(float) * batch_size * input_size);
    }
}

void Cross_Entropy_Loss::setTarget(float *target) { this->target = target; }