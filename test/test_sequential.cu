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

    int size = 700; 
    int output_size = 10;   
    int device = 1;
    float learning_rate = 0.01;


    // Create a Sequential model
    Sequential* model = new Sequential(size, output_size); // Example input size of 784 (28x28 image flattened) and output size of 10 (number of classes)

    cout << "Model created with input size: " << model->getInputSize() << " and output size: " << model->getOutputSize() << endl;
    model->addLayer(new Linear(size, 1024, 1));
    model->addLayer(new ReLU(1024, 1024, 1)); // Example layer with input size 784, output size 10, and batch size 32
    model->addLayer(new Linear(1024, 256, 1)); // Example layer with input size 784, output size 10, and batch size 32
    model->addLayer(new ReLU(256, 256, 1)); // Example layer with input size 784, output size 10, and batch size 32
    model->addLayer(new Linear(256, output_size, 1)); // Example layer with input size 784, output size 10, and batch size 32
    model->addLayer(new Softmax(output_size, output_size)); // Example layer with input size 784, output size 10, and batch size 32
    model->loadModel("/home/MORGRIDGE/akazi/HPC_Assignments/Final_Project/CNN_Implementation_on_CUDA/model_weights.txt");
    model->summary();

    // Loss Function
    Cross_Entropy_Loss* loss_fn = new Cross_Entropy_Loss(output_size, 1);


    if (device == 1)
    {
        // Set the device for the model
        model->setDevice(1); // Set the device to GPU (0 for CPU, 1 for GPU)
        loss_fn->setDevice(1); // Set the device to GPU (0 for CPU, 1 for GPU)
        cout << "Model device set to GPU" << endl;
    }


    for(int j = 0; j < 3; j++)
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

        // Groutd turth
        float* h_target = new float[1]{0};
        float* target;
        // Allocate memory for the target on the device
        if (device == 0)
        { 
            target = h_target;
        }
        else
        { 
            cudaMalloc((void**)&target, sizeof(float));
            cudaMemcpy(target, h_target, sizeof(float), cudaMemcpyHostToDevice);
        }


    


        // Forward Pass the Model
        float* output = model->forward(arr);

        // Move the output from device to host and print the values
        float* h_output = new float[output_size];
        if (device == 1)
        {
            cudaMemcpy(h_output, output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
        }
        else
        {
            h_output = output;
        }

        cout << "Output values: ";
        for (int i = 0; i < output_size; ++i)
        {
            cout << h_output[i] << " ";
        }
        cout << endl;

        // Free the host memory
        delete[] h_output;





        // Backward Pass the Model
        float loss_value = model->backward(output, target ,loss_fn);

        //Print the Loss
        cout << "Loss: " << loss_value << endl;

        // Update the weights
        model->updateParameters(learning_rate); // Example learning rate of 0.01

        // Move the output from device to host

 }
    

    return 0;
}


