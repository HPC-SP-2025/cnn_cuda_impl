#include "../../include/cnn_library/layers/cross_entropy_loss.h"
#include <cmath>
#include <iostream>
using namespace std;

#define BLOCK_SIZE 256
#define EPSILON 1e-9f

/**
 * output: Flattened 1D vector with class probabilites
 * target: class label 0-indexed
 */
float Cross_Entropy_Loss::forward_CPU(const float *pred, float *target) {
    int n = batch_size;
    int num_classes = input_size;

    float loss = 0.0;

#pragma omp parallel for reduction(- : loss)
    for (auto i = 0; i < n; i++) {
        loss -= log(max(pred[i * num_classes + (int)target[i]], EPSILON));
    }
    loss /= static_cast<float>(n);
    return loss;
}

void Cross_Entropy_Loss::backward_CPU(float *grad_output, float *pred, float *target) {

    int n = batch_size;
    int num_classes = input_size;

#pragma omp parallel for
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
}

float *Cross_Entropy_Loss::forward(float *pred) {

    float loss;

    loss = forward_CPU(pred, this->target);

    host_forward_buffer[0] = loss;
    return host_forward_buffer;
}

float *Cross_Entropy_Loss::backward(float *pred) {

    backward_CPU(host_backward_buffer, pred, this->target);
    return host_backward_buffer;
}

void Cross_Entropy_Loss::setDevice(int device) { this->device = device; }

void Cross_Entropy_Loss::setTarget(float *target) { this->target = target; }