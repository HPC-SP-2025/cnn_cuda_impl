#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../include/cnn_library/layers/linear.h"

#define EPSILON 1e-5

bool almost_equal(float a, float b, float epsilon = EPSILON) {
    return std::fabs(a - b) < epsilon;
}

// Test the forward pass with known weights and biases
void test_linear_forward_basic() {
    std::cout << "Running test_linear_forward_basic...\n";
    
    // Create a linear layer with 2 inputs and 3 outputs
    Linear linear_layer(2, 3, 1);
    
    // Manually set weights and biases for predictable output
    float weights[6] = {
        0.1f, 0.2f, 0.3f,  // weights for input 1
        0.4f, 0.5f, 0.6f   // weights for input 2
    };
    float biases[3] = {0.1f, 0.2f, 0.3f};
    
    // Set the weights and biases directly (requires exposing setWeights/setBiases methods)
    linear_layer.setWeights(weights);
    linear_layer.setBiases(biases);
    
    // Input data
    float input[2] = {1.0f, 2.0f};
    
    // Expected output calculation:
    // output[0] = 1.0*0.1 + 2.0*0.4 + 0.1 = 0.1 + 0.8 + 0.1 = 1.0
    // output[1] = 1.0*0.2 + 2.0*0.5 + 0.2 = 0.2 + 1.0 + 0.2 = 1.4
    // output[2] = 1.0*0.3 + 2.0*0.6 + 0.3 = 0.3 + 1.2 + 0.3 = 1.8
    float expected[3] = {1.0f, 1.4f, 1.8f};
    
    // Perform forward pass
    float *output = linear_layer.forward(input);
    
    // Verify results
    for (int i = 0; i < 3; ++i) {
        assert(almost_equal(output[i], expected[i]));
    }
    
    std::cout << "Passed.\n";
}

// Test the forward pass with batch processing
void test_linear_forward_batch() {
    std::cout << "Running test_linear_forward_batch...\n";
    
    // Create a linear layer with 2 inputs, 3 outputs, and batch size 2
    Linear linear_layer(2, 3, 2);
    
    // Manually set weights and biases
    float weights[6] = {
        0.1f, 0.2f, 0.3f,  // weights for input 1
        0.4f, 0.5f, 0.6f   // weights for input 2
    };
    float biases[3] = {0.1f, 0.2f, 0.3f};
    
    linear_layer.setWeights(weights);
    linear_layer.setBiases(biases);
    
    // Input data for batch of 2
    float input[4] = {
        1.0f, 2.0f,  // First example
        3.0f, 4.0f   // Second example
    };
    
    // Expected output for first example:
    // Same as in basic test: [1.0f, 1.4f, 1.8f]
    // Expected output for second example:
    // output[0] = 3.0*0.1 + 4.0*0.4 + 0.1 = 0.3 + 1.6 + 0.1 = 2.0
    // output[1] = 3.0*0.2 + 4.0*0.5 + 0.2 = 0.6 + 2.0 + 0.2 = 2.8
    // output[2] = 3.0*0.3 + 4.0*0.6 + 0.3 = 0.9 + 2.4 + 0.3 = 3.6
    float expected[6] = {
        1.0f, 1.4f, 1.8f,  // First example
        2.0f, 2.8f, 3.6f   // Second example
    };
    
    // Perform forward pass
    float *output = linear_layer.forward(input);
    
    // Verify results
    for (int i = 0; i < 6; ++i) {
        assert(almost_equal(output[i], expected[i]));
    }
    
    std::cout << "Passed.\n";
}

// Test the backward pass
void test_linear_backward() {
    std::cout << "Running test_linear_backward...\n";
    
    // Create a linear layer with 2 inputs and 3 outputs
    Linear linear_layer(2, 3, 1);
    
    // Manually set weights and biases
    float weights[6] = {
        0.1f, 0.2f, 0.3f,  // weights for input 1
        0.4f, 0.5f, 0.6f   // weights for input 2
    };
    float biases[3] = {0.1f, 0.2f, 0.3f};
    
    linear_layer.setWeights(weights);
    linear_layer.setBiases(biases);
    
    // Input data
    float input[2] = {1.0f, 2.0f};
    
    // Forward pass first to cache input
    float *output = linear_layer.forward(input);
    
    // Gradient coming from next layer
    float grad_input[3] = {1.0f, 1.0f, 1.0f};
    
    // Expected gradient with respect to input:
    // grad_output[0] = 1.0*0.1 + 1.0*0.2 + 1.0*0.3 = 0.6
    // grad_output[1] = 1.0*0.4 + 1.0*0.5 + 1.0*0.6 = 1.5
    float expected_grad_output[2] = {0.6f, 1.5f};
    
    // Expected gradients for weights and biases should also be calculated
    // but we're not testing them in this basic test
    
    // Perform backward pass
    float *grad_output = linear_layer.backward(grad_input);
    
    // Verify gradient with respect to input
    for (int i = 0; i < 2; ++i) {
        assert(almost_equal(grad_output[i], expected_grad_output[i]));
    }
    
    std::cout << "Passed.\n";
}

// Test parameter update functionality
void test_linear_parameter_update() {
    std::cout << "Running test_linear_parameter_update...\n";
    
    // Create a linear layer with 2 inputs and 2 outputs
    Linear linear_layer(2, 2, 1);
    
    // Manually set initial weights and biases
    float initial_weights[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float initial_biases[2] = {0.5f, 0.6f};
    
    linear_layer.setWeights(initial_weights);
    linear_layer.setBiases(initial_biases);
    
    // Forward pass with input
    float input[2] = {1.0f, 1.0f};
    float *output = linear_layer.forward(input);
    
    // Backward pass with gradient
    float grad_input[2] = {1.0f, 1.0f};
    float *grad_output = linear_layer.backward(grad_input);
    
    // Now the gradients should be computed
    // Weight gradients: input[i] * grad_input[j]
    // For batch size 1 and input [1,1], weight gradients should be [1,1,1,1]
    // Bias gradients: grad_input[j] = [1,1]
    
    // Update parameters with learning rate 0.1
    float learning_rate = 0.1f;
    linear_layer.updateParameters(learning_rate);
    
    // Expected updated weights:
    // weights[i,j] -= learning_rate * weight_gradients[i,j]
    // = initial_weights[i,j] - 0.1 * 1.0
    float expected_weights[4] = {0.0f, 0.1f, 0.2f, 0.3f};
    
    // Expected updated biases:
    // biases[j] -= learning_rate * bias_gradients[j]
    // = initial_biases[j] - 0.1 * 1.0
    float expected_biases[2] = {0.4f, 0.5f};
    
    // Get the updated weights and biases
    float updated_weights[4];
    float updated_biases[2];
    linear_layer.getWeights(updated_weights);
    linear_layer.getBiases(updated_biases);
    
    // Verify updated weights and biases
    for (int i = 0; i < 4; ++i) {
        assert(almost_equal(updated_weights[i], expected_weights[i]));
    }
    
    for (int i = 0; i < 2; ++i) {
        assert(almost_equal(updated_biases[i], expected_biases[i]));
    }
    
    std::cout << "Passed.\n";
}

// Test Xavier/Glorot initialization
void test_linear_weight_initialization() {
    std::cout << "Running test_linear_weight_initialization...\n";
    
    // Create a linear layer with significant size to test statistics
    size_t input_size = 100;
    size_t output_size = 50;
    Linear linear_layer(input_size, output_size, 1);
    
    // Get the initialized weights
    float* weights = new float[input_size * output_size];
    linear_layer.getWeights(weights);
    
    // Calculate mean and variance
    float sum = 0.0f;
    float sum_squared = 0.0f;
    size_t total_weights = input_size * output_size;
    
    for (size_t i = 0; i < total_weights; ++i) {
        sum += weights[i];
        sum_squared += weights[i] * weights[i];
    }
    
    float mean = sum / total_weights;
    float variance = (sum_squared / total_weights) - (mean * mean);
    
    // Xavier initialization should have mean close to 0 and 
    // variance close to 2 / (input_size + output_size)
    float expected_variance = 2.0f / (input_size + output_size);
    
    // Check that mean is close to 0 (allowing for some statistical deviation)
    assert(std::fabs(mean) < 0.05);
    
    // Check that variance is close to expected (allowing for some statistical deviation)
    assert(std::fabs(variance - expected_variance) < 0.01);
    
    delete[] weights;
    
    std::cout << "Passed.\n";
}

int main() {
    // Uncomment tests as you implement the corresponding functionality
    test_linear_forward_basic();
    test_linear_forward_batch();
    test_linear_backward();
    test_linear_parameter_update();
    test_linear_weight_initialization();
    
    std::cout << "All tests passed!\n";
    return 0;
}
