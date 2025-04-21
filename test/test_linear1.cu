#include <iostream>
#include <cassert>
#include <cmath>
#include "../include/cnn_library/layers/linear.h"

#define EPSILON 1e-5

bool almost_equal(float a, float b, float epsilon = EPSILON) {
    return std::fabs(a - b) < epsilon;
}

void test_constructor_cpu() {
    Linear layer(4, 3, 2);  // input:4, output:3, batch:2

    assert(layer.getInputSize() == 4);
    assert(layer.getOutputSize() == 3);
    //assert(layer.getBatchSize() == 2);

    std::cout << "Test Constructor (CPU) Passed\n";
}

void test_forward_cpu() {
    Linear layer(2, 2, 1);  // simple 2x2 layer

    float weights[] = {1, 2, 3, 4};   // [ [1, 2], [3, 4] ]
    float biases[] = {0.5, -0.5};
    float input[] = {1, 2};           // input: [1, 2]
    float expected[] = {1*1 + 2*3 + 0.5, 1*2 + 2*4 - 0.5}; // [7.5, 9.5]

    layer.setWeights(weights);
    layer.setBiases(biases);

    float* output = layer.forward(input);
    assert(almost_equal(output[0], expected[0]));
    assert(almost_equal(output[1], expected[1]));

    std::cout << "Test Forward (CPU) Passed\n";
}

void test_weights_bias_io() {
    Linear layer(2, 2, 1);

    float weights[] = {1, 2, 3, 4};
    float biases[] = {0.1f, -0.1f};
    float w_read[4], b_read[2];

    layer.setWeights(weights);
    layer.setBiases(biases);

    layer.getWeights(w_read);
    layer.getBiases(b_read);

    for (int i = 0; i < 4; ++i) assert(weights[i] == w_read[i]);
    for (int i = 0; i < 2; ++i) assert(biases[i] == b_read[i]);

    std::cout << "Test Set/Get Weights & Biases Passed\n";
}

void test_forward_gpu() {
    // Create layer
    Linear layer(2, 2, 1);  // input_size=2, output_size=2, batch_size=1
    layer.setDevice(1); // enable GPU

    // Prepare weights and biases
    float weights[] = {1, 2, 3, 4}; // 2x2 matrix: [[1,2], [3,4]]
    float biases[] = {0.5, -0.5};   // 2 biases
    layer.setWeights(weights);
    layer.setBiases(biases);

    // Allocate input on device
    float input_host[] = {1.0, 2.0};  // 1x2 input
    float *input_device;
    cudaMalloc(&input_device, sizeof(input_host));
    cudaMemcpy(input_device, input_host, sizeof(input_host), cudaMemcpyHostToDevice);

    // Run GPU forward
    float *output_device = layer.forward(input_device);

    // Copy result back to host
    float output_host[2];
    cudaMemcpy(output_host, output_device, sizeof(output_host), cudaMemcpyDeviceToHost);

    // Expected: y = xW + b = [1*1+2*3+0.5, 1*2+2*4-0.5] = [7.5, 9.5]
    assert(almost_equal(output_host[0], 7.5));
    assert(almost_equal(output_host[1], 9.5));

    std::cout << "Test Forward (GPU) Passed\n";

    // Cleanup
    cudaFree(input_device);
}

void test_backward_cpu() {
    Linear layer(2, 2, 1);  // input:2, output:2, batch:1

    // Set known weights and input
    float weights[] = {1, 2, 3, 4};   // 2x2
    float input[] = {1, 2};           // 1x2
    float grad_out[] = {0.5, -1.0};   // dL/dy: 1x2

    layer.setWeights(weights);
    layer.forward(input);  // cache input
    float* grad_input = layer.backward(grad_out);  // calls backwardCPU()

    // Expected: grad_input = grad_out * Wᵗ = [0.5, -1] * [[1,3],[2,4]]ᵗ = [0.5*1 + -1*2, 0.5*3 + -1*4] = [-1.5, -2.5]
    assert(almost_equal(grad_input[0], -1.5f));
    assert(almost_equal(grad_input[1], -2.5f));

    // Check gradients
    float expected_dW[] = {0.5*1, 1*(-1.0), 0.5*2, -1.0*2};  // xᵗ * grad_out
    float expected_db[] = {0.5f, -1.0f};

    for (int i = 0; i < 4; i++) assert(almost_equal(layer.host_grad_weights[i], expected_dW[i]));
    for (int i = 0; i < 2; i++) assert(almost_equal(layer.host_grad_biases[i], expected_db[i]));

    std::cout << "Test Backward (CPU) Passed\n";
}
void test_backward_gpu() {
    Linear layer(2, 2, 1);
    layer.setDevice(1);

    float weights[] = {1, 2, 3, 4};
    float input_host[] = {1.0f, 2.0f};
    float grad_out_host[] = {0.5f, -1.0f};

    layer.setWeights(weights);
    
    // Allocate and copy input/grad_out to device
    float *input_dev, *grad_out_dev;
    cudaMalloc(&input_dev, sizeof(input_host));
    cudaMalloc(&grad_out_dev, sizeof(grad_out_host));
    cudaMemcpy(input_dev, input_host, sizeof(input_host), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_out_dev, grad_out_host, sizeof(grad_out_host), cudaMemcpyHostToDevice);

    layer.forward(input_dev);  // GPU forward caches device input
    float* grad_input_dev = layer.backward(grad_out_dev);  // GPU backward

    // Copy grad_input back
    float grad_input_host[2];
    cudaMemcpy(grad_input_host, grad_input_dev, sizeof(grad_input_host), cudaMemcpyDeviceToHost);

    // Validate grad_input
    assert(almost_equal(grad_input_host[0], -1.5));
    assert(almost_equal(grad_input_host[1], -2.5));

    // Copy gradients
    float grad_w[4], grad_b[2];
    cudaMemcpy(grad_w, layer.device_grad_weights, sizeof(grad_w), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_b, layer.device_grad_biases, sizeof(grad_b), cudaMemcpyDeviceToHost);

    float expected_dW[] = {0.5*1, 1*(-1.0), 0.5*2, -1.0*2};  // xᵗ * grad_out
    float expected_db[] = {0.5f, -1.0f};

    for (int i = 0; i < 4; i++) assert(almost_equal(grad_w[i], expected_dW[i]));
    for (int i = 0; i < 2; i++) assert(almost_equal(grad_b[i], expected_db[i]));

    std::cout << "Test Backward (GPU) Passed\n";

    cudaFree(input_dev);
    cudaFree(grad_out_dev);
}

void test_update_parameters() {
    Linear layer(2, 2, 1);  // input:2, output:2, batch:1

    // Set initial weights and biases
    float weights[] = {1.0f, 2.0f, 3.0f, 4.0f};  // 2x2
    float biases[] = {0.5f, -0.5f};
    layer.setWeights(weights);
    layer.setBiases(biases);

    // Manually set gradients (dW, db)
    float grad_weights[] = {0.1f, 0.2f, 0.3f, 0.4f};
    float grad_biases[] = {0.05f, -0.05f};
    std::memcpy(layer.host_grad_weights, grad_weights, sizeof(grad_weights));
    std::memcpy(layer.host_grad_biases, grad_biases, sizeof(grad_biases));

    // Update with lr = 0.1
    layer.updateParameters(0.1f);

    // Expected weights: w - lr * dw
    float expected_weights[] = {
        1.0f - 0.1f * 0.1f,  // 0.99
        2.0f - 0.1f * 0.2f,  // 1.98
        3.0f - 0.1f * 0.3f,  // 2.97
        4.0f - 0.1f * 0.4f   // 3.96
    };
    float expected_biases[] = {
        0.5f - 0.1f * 0.05f,   // 0.495
        -0.5f - 0.1f * -0.05f  // -0.495
    };

    float new_weights[4], new_biases[2];
    layer.getWeights(new_weights);
    layer.getBiases(new_biases);

    for (int i = 0; i < 4; ++i) {
        assert(almost_equal(new_weights[i], expected_weights[i]));
    }
    for (int i = 0; i < 2; ++i) {
        assert(almost_equal(new_biases[i], expected_biases[i]));
    }

    std::cout << "Test UpdateParameters Passed\n";
}

int main() {
  test_constructor_cpu();
  test_weights_bias_io();
  test_forward_cpu();
  test_forward_gpu();
  test_backward_cpu();
  test_backward_gpu();
  test_update_parameters();

  std::cout << "All Linear Tests passed!\n";
}
