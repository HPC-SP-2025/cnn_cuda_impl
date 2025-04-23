#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../include/cnn_library/layers/softmax.h"

#define EPSILON 1e-5

bool almost_equal(float a, float b, float epsilon = EPSILON) {
    return std::fabs(a - b) < epsilon;
}

void test_softmax_forward_cpu_basic_case() {
    std::cout << "Running test_softmax_forward_cpu_basic_case...\n";

    Softmax softmax_layer(3, 1); // 3 classes, batch size 1

    float input[3] = {1.0f, 2.0f, 3.0f};
    float* output = softmax_layer.forward(input);

    // Expected softmax output
    float sum = std::exp(1.0f - 3.0f) + std::exp(2.0f - 3.0f) + std::exp(3.0f - 3.0f);
    float expected[3] = {
        std::exp(1.0f - 3.0f) / sum,
        std::exp(2.0f - 3.0f) / sum,
        std::exp(3.0f - 3.0f) / sum,
    };

    for (int i = 0; i < 3; ++i) {
        assert(almost_equal(output[i], expected[i]));
    }

    std::cout << "Passed.\n";
}

void test_softmax_forward_cpu_batch_case() {
    std::cout << "Running test_softmax_forward_cpu_batch_case...\n";

    Softmax softmax_layer(2, 2); // 2 classes, batch size 2

    float input[4] = {
        1.0f, 2.0f, // First example
        2.0f, 1.0f  // Second example
    };

    float* output = softmax_layer.forward(input);

    for (int b = 0; b < 2; ++b) {
        float max_val = std::max(input[b * 2], input[b * 2 + 1]);
        float sum = std::exp(input[b * 2] - max_val) + std::exp(input[b * 2 + 1] - max_val);
        float expected0 = std::exp(input[b * 2] - max_val) / sum;
        float expected1 = std::exp(input[b * 2 + 1] - max_val) / sum;

        assert(almost_equal(output[b * 2], expected0));
        assert(almost_equal(output[b * 2 + 1], expected1));
    }

    std::cout << "Passed.\n";
}

void test_softmax_forward_cpu_numerical_stability() {
    std::cout << "Running test_softmax_forward_cpu_numerical_stability...\n";

    Softmax softmax_layer(2, 1);

    float input[2] = {1000.0f, 1000.0f}; // Large identical values
    float* output = softmax_layer.forward(input);

    assert(almost_equal(output[0], 0.5f));
    assert(almost_equal(output[1], 0.5f));

    std::cout << "Passed.\n";
}

void test_softmax_backward_cpu_basic_case() {
    std::cout << "Running test_softmax_backward_cpu_basic_case...\n";

    Softmax softmax_layer(3, 1); // 3 classes, batch size 1

    float input[3] = {1.0f, 2.0f, 3.0f};
    float* output = softmax_layer.backward(input);

    // Expected softmax output
    float expected[3] = {1.0f, 2.0f, 3.0f};

    for (int i = 0; i < 3; ++i) {
        assert(almost_equal(output[i], expected[i]));
    }

    std::cout << "Passed.\n";
}

void test_softmax_forward_gpu_basic_case() {
    std::cout << "Running test_softmax_forward_gpu_basic_case...\n";

    Softmax softmax_layer(3, 1); // 3 classes, batch size 1

    softmax_layer.setDevice(1);
    softmax_layer.setTestMode(1);

    float input[3] = {1.0f, 2.0f, 3.0f};
    float* output = softmax_layer.forward(input);

    // Expected softmax output
    float sum = std::exp(1.0f - 3.0f) + std::exp(2.0f - 3.0f) + std::exp(3.0f - 3.0f);
    float expected[3] = {
        std::exp(1.0f - 3.0f) / sum,
        std::exp(2.0f - 3.0f) / sum,
        std::exp(3.0f - 3.0f) / sum,
    };

    for (int i = 0; i < 3; ++i) {
        assert(almost_equal(output[i], expected[i]));
    }

    std::cout << "Passed.\n";
}

void test_softmax_forward_gpu_batch_case() {
    std::cout << "Running test_softmax_forward_gpu_batch_case...\n";

    Softmax softmax_layer(2, 2); // 2 classes, batch size 2
    softmax_layer.setDevice(1);
    softmax_layer.setTestMode(1);

    float input[4] = {
        1.0f, 2.0f, // First example
        2.0f, 1.0f  // Second example
    };

    float* output = softmax_layer.forward(input);

    for (int b = 0; b < 2; ++b) {
        float max_val = std::max(input[b * 2], input[b * 2 + 1]);
        float sum = std::exp(input[b * 2] - max_val) + std::exp(input[b * 2 + 1] - max_val);
        float expected0 = std::exp(input[b * 2] - max_val) / sum;
        float expected1 = std::exp(input[b * 2 + 1] - max_val) / sum;

        assert(almost_equal(output[b * 2], expected0));
        assert(almost_equal(output[b * 2 + 1], expected1));
    }

    std::cout << "Passed.\n";
}

void test_softmax_forward_gpu_numerical_stability() {
    std::cout << "Running test_softmax_forward_gpu_numerical_stability...\n";

    Softmax softmax_layer(2, 1);
    softmax_layer.setDevice(1);
    softmax_layer.setTestMode(1);

    float input[2] = {1000.0f, 1000.0f}; // Large identical values
    float* output = softmax_layer.forward(input);

    assert(almost_equal(output[0], 0.5f));
    assert(almost_equal(output[1], 0.5f));

    std::cout << "Passed.\n";
}

void test_softmax_backward_gpu_basic_case() {
    std::cout << "Running test_softmax_backward_gpu_basic_case...\n";

    Softmax softmax_layer(3, 1); // 3 classes, batch size 1
    softmax_layer.setDevice(1);
    softmax_layer.setTestMode(1);

    float input[3] = {1.0f, 2.0f, 3.0f};
    float* output = softmax_layer.backward(input);

    // Expected softmax output
    float expected[3] = {1.0f, 2.0f, 3.0f};

    for (int i = 0; i < 3; ++i) {
        assert(almost_equal(output[i], expected[i]));
    }

    std::cout << "Passed.\n";
}

int main() {
    test_softmax_forward_cpu_basic_case();
    test_softmax_forward_cpu_batch_case();
    test_softmax_forward_cpu_numerical_stability();
    test_softmax_backward_cpu_basic_case();
    test_softmax_forward_gpu_basic_case();
    test_softmax_forward_gpu_batch_case();
    test_softmax_forward_gpu_numerical_stability();
    test_softmax_backward_gpu_basic_case();

    std::cout << "All tests passed.\n";
    return 0;
}
