# include <iostream>
# include <vector>
#include "model.h" 
# include "cnn_library/layers/loss.h"
using namespace std;



int main()
{ 

    // Hyperparameters
    int batch_size = 64;
    float lr = 0.001;    
    int total_iterations = 10000;
    int device = 0; // 0 for CPU, 1 for GPU
    int input_size = 32;
    int num_classes = 10;

    // Create the MNIST model
    Sequential* net = create_mnist_model();
    net->setDevice(device);

    // Create the loss layer
    Loss* loss_layer = new Loss();


    // Create the training for loop For Epoch
    std::cout << "Training on " << device << " for " << total_iterations << " iterations." << std::endl;

    // Iterate over the the data
    for (int iter = 0; iter < total_iterations; iter++)
    {
        
        // Create a random array for input data and labels ( Will be replaced with dataloader)
        std::vector<float> input_data(batch_size * input_size);
        std::vector<float> label_vector(batch_size * num_classes);
        std::vector<float> prediction(batch_size * input_size);


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