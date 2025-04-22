#include <iostream>
#include "../include/cnn_library/nn/sequential.h"
#include "../include/cnn_library/layers/relu.h"
#include "../include/cnn_library/layers/softmax.h"
#include "../include/cnn_library/layers/cross_entropy_loss.h"
#include "../include/cnn_library/layers/linear.h"



using namespace std;

// Function to create and return a random array with random positive or negative numbers
#include <cstdlib>
#include <ctime>

float* createAndReturnRandomArray(unsigned int size) 
{
    const float constantValue = 10.0f; // Constant value to fill the array
    float* constantArray = new float[size];
    for (unsigned int i = 0; i < size; ++i) {
        constantArray[i] = constantValue;
    }
    return constantArray;
}

int main(){

    int size = 5;    
    int device = 1;


    // Create a Sequential model
    Sequential* model = new Sequential(5, 1); // Example input size of 784 (28x28 image flattened) and output size of 10 (number of classes)

    cout << "Model created with input size: " << model->getInputSize() << " and output size: " << model->getOutputSize() << endl;
    model->addLayer(new Linear(5, 128, 1));
    model->addLayer(new ReLU(128, 128, 1)); // Example layer with input size 784, output size 10, and batch size 32
    model->addLayer(new Linear(128, 1, 1)); // Example layer with input size 784, output size 10, and batch size 32
    model->loadModel("/home/MORGRIDGE/akazi/HPC_Assignments/Final_Project/CNN_Implementation_on_CUDA/model_weights.txt");

    // LOSS FUNCTION
    // Cross_Entropy_Loss* loss_fn = new Cross_Entropy_Loss(5, 1); // Example layer with input size 784, output size 10, and batch size 32
    // loss_fn->setTarget(new float[1]{4}); // Set the target for the loss function
    // loss_fn->setDevice(1); // Set the device to CPU (0 for CPU, 1 for GPU)


    if (device == 1)
    {
        // Set the device for the model
        model->setDevice(1); // Set the device to GPU (0 for CPU, 1 for GPU)
        cout << "Model device set to GPU" << endl;
    }


    
    
    
    for(int j = 0; j < 1; j++)
    {
        cout << "Iteration: " << j << endl;

        // Input Image
        float* h_arr = createAndReturnRandomArray(size);
        float* arr;
        

        // Allocate memory on the device
        if (device == 0)
        { 
            arr = h_arr;
        }
        else if (device == 1)
        { 
            cudaMalloc((void**)&arr, size * sizeof(float));
            cudaMemcpy(arr, h_arr, size * sizeof(float), cudaMemcpyHostToDevice);

        }

        // Grouth Truth
        // int targetIndex = 0;
        // for (int i = 1; i < size; ++i) 
        // {
        //     if (arr[i] > arr[targetIndex]) 
        //     {
        //         targetIndex = i;
        //     }
        // }

        // loss_fn->setTarget(new float[1]{static_cast<float>(targetIndex)}); // Set the target for the loss function
        // float* arr = new float[size]{-1.0, -.05, 0, 0, 0.8}; // Example input array
        cout << "Image Pointer: " << arr << endl;

        // Add layers to the model
        float* output = model->forward(arr);
        // float* loss_value = loss_fn->forward(output);
        // cout << "Loss Value: " << *loss_value << endl;

        // Move the output from device to host
        float* h_output = new float[model->getOutputSize()];
        if (device == 1)
        {   
            cudaMemcpy(h_output, output, model->getOutputSize() * sizeof(float), cudaMemcpyDeviceToHost);
        }

        else if (device == 0)
        {   
            h_output = output;
        }

  
        // Use h_output for further processing if needed
        

        // // Print the array
        // cout << "";
        // for (int i = 0; i < size; ++i) {
        //     cout << arr[i] << " ";
        //     // printf("%f ", arr[i]);
        // }
        // cout << endl;

        cout << endl;
        cout << "";
        for (int i = 0; i < model->getOutputSize(); ++i) {
            cout << h_output[i] << " ";
        }
        cout << endl;

 }
    

    return 0;
}


