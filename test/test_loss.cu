#include "../include/cnn_library/layers/cross_entropy_loss.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

bool float_equals(float a, float b, float epsilon = 1e-5) { return std::fabs(a - b) < epsilon; }
bool array_equals(float *a, float *b, size_t n, float epsilon = 1e-5) {

    for (size_t i = 0; i < n; i++)
        if (!float_equals(a[i], b[i], epsilon))
            return false;

    return true;
}

void test_forward_cpu() {
    printf("Running CrossEntropyLoss forward (CPU)...\n");
    Cross_Entropy_Loss loss(4, 1);

    loss.setDevice(0);

    float predictions[] = {0.25, 0.25, 0.25, 0.25};
    float targets[] = {3};

    loss.setTarget(targets);
    float *out = loss.forward(predictions);
    float expected = -log(0.25);
    assert(float_equals(out[0], expected));
    printf("PASSED\n");
}

void test_backward_cpu() {
    printf("Running CrossEntropyLoss backward (CPU)...\n");

    Cross_Entropy_Loss loss(4, 1);
    loss.setDevice(0);

    float predictions[] = {0.25, 0.25, 0.25, 0.25};
    float targets[] = {3};

    loss.setTarget(targets);
    float *grads = loss.backward(predictions);

    float expected[] = {0.25, 0.25, 0.25, -0.75};

    assert(array_equals(grads, expected, 4));
    printf("PASSED\n");
}

void test_forward_gpu() {
    printf("Running CrossEntropyLoss forward (GPU)...\n");
    Cross_Entropy_Loss loss(4, 1);

    loss.setDevice(1);

    float h_preds[] = {0.25, 0.25, 0.25, 0.25};
    float h_target[] = {3};

    float *d_preds, *d_targets;
    cudaMalloc(&d_preds, 4 * sizeof(float));
    cudaMalloc(&d_targets, 4 * sizeof(float));
    cudaMemcpy(d_preds, h_preds, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_target, 4 * sizeof(float), cudaMemcpyHostToDevice);

    loss.setTarget(d_targets);
    float *d_out = loss.forward(d_preds);
    float expected = -log(0.25);

    float *out = new float[1];
    cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    assert(float_equals(out[0], expected));
    printf("PASSED\n");
}

void test_backward_gpu() {
    printf("Running CrossEntropyLoss backward (GPU)...\n");

    Cross_Entropy_Loss loss(4, 1);
    loss.setDevice(1);

    float h_preds[] = {0.25, 0.25, 0.25, 0.25};
    float h_target[] = {3};

    float *d_preds, *d_targets;
    cudaMalloc(&d_preds, 4 * sizeof(float));
    cudaMalloc(&d_targets, 4 * sizeof(float));
    cudaMemcpy(d_preds, h_preds, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_target, 4 * sizeof(float), cudaMemcpyHostToDevice);

    loss.setTarget(d_targets);
    float *d_grads = loss.backward(d_preds);

    float *grads = new float[4];
    cudaMemcpy(grads, d_grads, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    float expected[] = {0.25, 0.25, 0.25, -0.75};
    assert(array_equals(grads, expected, 4));
    printf("PASSED\n");
}

int main() {
    printf("Running Cross Entropy Loss Test cases\n");
    test_forward_cpu();
    test_backward_cpu();

    test_forward_gpu();
    test_backward_gpu();

    printf("All test Passed!\n");
    return 0;
}