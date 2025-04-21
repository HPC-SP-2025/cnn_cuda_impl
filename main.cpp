#include "include/cnn_library/layers/cross_entropy_loss.h"
#include "include/cnn_library/layers/loss.h"
#include "model.h"
#include <iostream>
#include <vector>
using namespace std;

int main() {

    // Hyperparameters
    int batch_size = 64;
    float lr = 0.001;
    int total_iterations = 10000;
    int device = 0; // 0 for CPU, 1 for GPU
    int input_size = 768;
    int num_classes = 10;

    // Create the MNIST model
    Sequential *net = create_mnist_model(input_size = input_size, batch_size = batch_size, num_classes = num_classes);

    // Print the model summary
    net->summary();

    // Set the device for the model
    net->setDevice(device);

    // Create the loss layer
    Loss *loss_layer = new Cross_Entropy_Loss(num_classes, batch_size);

    // Create the training for loop For Epoch
    std::cout << "Training on " << device << " for " << total_iterations << " iterations." << std::endl;

    // Iterate over the the data
    for (int iter = 0; iter < total_iterations; iter++) {

        // Create a random array for input data and labels ( Will be replaced with dataloader)
        float *input_data = new float[batch_size * input_size * input_size];
        float *label_vector = new float[batch_size * num_classes];
        float *prediction = new float[batch_size * num_classes];

        // Fill the input data and labels with random values
        for (int i = 0; i < batch_size * input_size * input_size; i++) {
            input_data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }

        // Fill the label vector with random values
        for (int i = 0; i < batch_size * num_classes; i++) {
            label_vector[i] = static_cast<float>(rand()) / static_cast<int>(RAND_MAX);
        }

        // Forwardfeed the model
        net->forward(input_data, prediction);

        // Backpropagation
        float loss_value = net->backward(prediction, label_vector, loss_layer);

        // Update the weights
        net->updateParameters(lr);

        // Print the loss
        std::cout << "Iteration: " << iter << ", Loss: " << loss_value << std::endl;
    }
}