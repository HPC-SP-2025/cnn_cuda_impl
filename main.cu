#include "include/cnn_library/layers/cross_entropy_loss.h"
#include "include/cnn_library/layers/loss.h"
#include "include/cnn_library/dataloader/dataloader.h"
#include "model.h"
#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

float* createAndReturnRandomArray(unsigned int size, float value) 
{
    const float constantValue = value; // Constant value to fill the array
    float* constantArray = new float[size];
    for (unsigned int i = 0; i < size; ++i) {
        constantArray[i] = constantValue;
    }
    return constantArray;
}

void saveVectorToFile(const std::vector<float>& vec, const std::string& filename) {
    std::ofstream outFile(filename);
    
    if (!outFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    
    for (const auto& value : vec) {
        outFile << value << "\n";  // Each value on a new line
    }
    
    outFile.close();
    std::cout << "Vector saved to " << filename << std::endl;
}



int main() {

    // HYPERPARAMETERS
    unsigned int batch_size = 1;
    unsigned int acc_batch_size = 128;
    float lr = 0.00001;
    int total_iterations = 50000;
    int device = 1; // 0 for CPU, 1 for GPU
    unsigned int input_size = 784;
    unsigned int num_classes = 10;

    string DATA_DIR = "/home/MORGRIDGE/akazi/HPC_Assignments/Final_Project/CNN_Implementation_on_CUDA/MNIST_Dataset/train-images-idx3-ubyte";
    string LABEL_DIR = "/home/MORGRIDGE/akazi/HPC_Assignments/Final_Project/CNN_Implementation_on_CUDA/MNIST_Dataset/train-labels-idx1-ubyte";


    // Create the MNIST model
    Sequential *net = create_mnist_model(input_size, batch_size, num_classes);

    // Create a Loss Layer
    Cross_Entropy_Loss *loss_layer = new Cross_Entropy_Loss(num_classes, batch_size);

    // Dataloader
    DataLoader *dataloader = new DataLoader(DATA_DIR, LABEL_DIR, batch_size, 60000);

    
    // Set the device for the model
    if (device == 1)
    { 
        net->setDevice(1);
        loss_layer->setDevice(1);
    }

    // Print the model summary
    net->summary();


    // Create the training for loop For Epoch
    std::cout << "Training on " << device << " for " << total_iterations << " iterations." << std::endl;

    // Create a random array for input data and labels ( Will be replaced with dataloader)
    float *h_input_data = new float[batch_size * input_size];
    float *h_label_vector = new float[batch_size];
    float *input_data;;
    float *label_vector;
    std::vector<float> loss_value_array;
    std::vector<float> iteration_wise_loss_array;

    // Iterate over the the data
    for (int iter = 0; iter < total_iterations; iter++) {

        // Fill the input data and labels with random values
        Batch batch_data = dataloader->load_batch(iter % 60000);
        h_input_data = batch_data.images;
        h_label_vector = batch_data.labels;
        

        if (device == 0) 
        {
            input_data = h_input_data;
            label_vector = h_label_vector;
        } 
        else 
        {
            cudaMalloc((void**)&input_data, input_size * batch_size * sizeof(float));
            cudaMemcpy(input_data, h_input_data, input_size * batch_size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&label_vector, 1 * batch_size * sizeof(float));
            cudaMemcpy(label_vector, h_label_vector, 1 * batch_size * sizeof(float), cudaMemcpyHostToDevice);
        }

        // Forwardfeed the model
        float* prediction = net->forward(input_data);

        // Backpropagation
        float loss_value = net->backward(prediction, label_vector, loss_layer);

        // Update the weights
        net->updateParameters(lr);

        // Print the loss
        if (iter % acc_batch_size == 0 && iter != 0) {
            float mean_loss = 0.0f;
            for (const auto& loss : loss_value_array) 
            {
            mean_loss += loss;
            }
            mean_loss =  mean_loss/loss_value_array.size();
            std::cout << "Iteration: " << iter << ", Mean Loss (last " << acc_batch_size << " iterations): " << mean_loss << std::endl;
            loss_value_array.clear(); // Reset the loss_value_array
            iteration_wise_loss_array.push_back(mean_loss);
        }
        else
        {
            loss_value_array.push_back(loss_value);
        }
        
    }
    saveVectorToFile(iteration_wise_loss_array, "iteration_wise_loss_values.txt");


    // Free the allocated memory
    delete[] h_input_data;
    delete[] h_label_vector;
    cudaFree(input_data);
    cudaFree(label_vector);
    delete net;
    delete loss_layer;
    return 0;
}