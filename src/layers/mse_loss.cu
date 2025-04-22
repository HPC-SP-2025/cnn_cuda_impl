#include "include/cnn_library/layers/mse_loss.h"
#include <stdexcept>
#include <vector>

#define BLOCK_SIZE 256

MSE_Loss::MSE_Loss(size_t batch_size) {
    this->layer_name = "MSE_Loss";
    this->batch_size = batch_size;

    this->host_forward_buffer = new float[1];
    this->host_backward_buffer = new float[batch_size];
}

MSE_Loss::~MSE_Loss() {
    delete[] host_backward_buffer;
    delete[] host_forward_buffer;
    if (device) {
        cudaFree(d_loss);
        cudaFree(device_backward_buffer);
    }
}

float *MSE_Loss::forward(float *pred) {
    float loss;
    if (this->device) {
        cudaMemset(d_loss, 0, sizeof(float));
        int num_blocks = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        forward_mse_kernel<<<num_blocks, BLOCK_SIZE>>>(pred, this->target, d_loss, batch_size);
        cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        loss = forward_CPU(pred, this->target);
    }
    host_forward_buffer[0] = loss;
    return host_forward_buffer;
}

float *MSE_Loss::backward(float *pred) {
    if (this->device) {
        int num_blocks = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        backward_mse_kernel<<<num_blocks, BLOCK_SIZE>>>(device_backward_buffer, pred, target, batch_size);
        return device_backward_buffer;
    } else {
        backward_CPU(host_backward_buffer, pred, target);
        return host_backward_buffer;
    }
}

void MSE_Loss::setDevice(int device) {
    this->device = device;
    if (device) {
        cudaMalloc(&d_loss, sizeof(float));
        cudaMalloc(&device_backward_buffer, sizeof(float) * batch_size);
    }
}
void MSE_Loss::setTarget(float *target) { this->target = target; }

float MSE_Loss::forward_CPU(const float *pred, float *target) {
    int n = batch_size;

    float loss = 0.0;
    for (auto i = 0; i < n; i++) {
        loss += pow(pred[i] - target[i], 2);
    }
    loss /= static_cast<float>(n);
    return loss;
}

// TODO: use reduction to avoid atomic adds
__global__ void forward_mse_kernel(const float *pred, float *target, float *loss, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float diff = pred[idx] - target[idx];
        atomicAdd(loss, diff * diff / n);
    }
}

void MSE_Loss::backward_CPU(float *grad_output, float *pred, float *target) {
    int n = batch_size;

    for (auto i = 0; i < n; i++) {
        grad_output[i] = 2.0 * (pred[i] - target[i]) / n;
    }
}

__global__ void backward_mse_kernel(float *grad_output, float *pred, float *target, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        grad_output[idx] = 2.0 * (pred[idx] - target[idx]) / n;
    }
}