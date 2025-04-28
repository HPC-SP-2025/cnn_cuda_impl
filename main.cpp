#include "include/cnn_library/dataloader/dataloader.h"
#include "include/cnn_library/layers/cross_entropy_loss.h"
#include "include/cnn_library/layers/loss.h"
#include "model.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

float *createAndReturnRandomArray(unsigned int size, float value) {
    const float constantValue = value; // Constant value to fill the array
    float *constantArray = new float[size];
    for (unsigned int i = 0; i < size; ++i) {
        constantArray[i] = constantValue;
    }
    return constantArray;
}

void saveVectorToFile(const std::vector<float> &vec, const std::string &filename) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (const auto &value : vec) {
        outFile << value << "\n"; // Each value on a new line
    }

    outFile.close();
    std::cout << "Vector saved to " << filename << std::endl;
}

int main(int argc, char const *argv[]) {

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    // HYPERPARAMETERS
    unsigned int batch_size = atoi(argv[1]);
    float lr = atof(argv[2]);
    int total_iterations = atoi(argv[3]);
    int device = atoi(argv[4]); // 0 for CPU, 1 for GPU
    unsigned int acc_batch_size = 10;
    unsigned int input_size = 784;
    unsigned int num_classes = 10;

    printf("batch size: %d, lr: %f, iters: %d, device: %d\n", batch_size, lr, total_iterations, device);

    if (device == 0) {
        int num_threads = atoi(argv[5]);
        omp_set_num_threads(num_threads);
#pragma omp parallel
#pragma omp single
        { printf("Num threads: %d\n", omp_get_num_threads()); }
    }

    string DATA_DIR = "../MNIST_Dataset/train-images-idx3-ubyte";
    string LABEL_DIR = "../MNIST_Dataset/train-labels-idx1-ubyte";

    // Create the MNIST model
    Sequential *net = create_mnist_model(input_size, batch_size, num_classes);

    // Create a Loss Layer
    Cross_Entropy_Loss *loss_layer = new Cross_Entropy_Loss(num_classes, batch_size);

    // Dataloader
    DataLoader *dataloader = new DataLoader(DATA_DIR, LABEL_DIR, batch_size, 60000);

    // Set the device for the model
    if (device == 1) {
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
    float *input_data;
    float *label_vector;
    std::vector<float> loss_value_array;
    std::vector<float> iteration_wise_loss_array;

    printf("batch size: %d, lr: %f, iters: %d, device:%d\n", batch_size, lr, total_iterations, device);

    int max_idx = 60000 / batch_size;

    start = high_resolution_clock::now();

    // Iterate over the the data
    for (int iter = 0; iter < total_iterations; iter++) {

        // Fill the input data and labels with random values
        Batch batch_data = dataloader->load_batch(iter % max_idx);
        input_data = batch_data.images;
        label_vector = batch_data.labels;

        // if (device == 0) {
        //     input_data = h_input_data;
        //     label_vector = h_label_vector;
        // }

        // Forwardfeed the model
        float *prediction = net->forward(input_data);

        // Backpropagation
        float loss_value = net->backward(prediction, label_vector, loss_layer);

        // Update the weights
        net->updateParameters(lr);

        // Print the loss
        if (iter % acc_batch_size == 0 && iter != 0) {
            float mean_loss = 0.0f;
            for (const auto &loss : loss_value_array) {
                mean_loss += loss;
            }
            mean_loss = mean_loss / loss_value_array.size();
            std::cout << "Iteration: " << iter << ", Mean Loss (last " << acc_batch_size
                      << " iterations): " << mean_loss << std::endl;
            loss_value_array.clear(); // Reset the loss_value_array
            iteration_wise_loss_array.push_back(mean_loss);
        } else {
            loss_value_array.push_back(loss_value);
        }
    }

    end = high_resolution_clock::now();
    duration_sec = chrono::duration_cast<duration<double, std::milli>>(end - start);
    printf("Total time: %fms, Per iter time: %fms\n", duration_sec.count(), duration_sec.count() / total_iterations);

    saveVectorToFile(iteration_wise_loss_array, "iteration_wise_loss_values.txt");

    // Free the allocated memory
    delete[] h_input_data;
    delete[] h_label_vector;
    delete net;
    delete loss_layer;
    return 0;
}