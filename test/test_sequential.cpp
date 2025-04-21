#include <iostream>
#include "../include/cnn_library/nn/sequential.h"
#include "../include/cnn_library/layers/relu.h"
#include "../include/cnn_library/layers/softmax.h"

using namespace std;

// Function to create and return a random array with random positive or negative numbers
#include <cstdlib>
#include <ctime>

float* createAndReturnRandomArray(unsigned int size) 
{
    srand(static_cast<unsigned int>(time(0))); // Seed the random number generator
    float* randomArray = new float[size];
    for (unsigned int i = 0; i < size; ++i) {
        randomArray[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // Random values between -1.0 and 1.0
    }
    return randomArray;
}

int main(){

    int size = 10;    


    // Create a Sequential model
    Sequential* model = new Sequential(size, size); // Example input size of 784 (28x28 image flattened) and output size of 10 (number of classes)
    cout << "Model created with input size: " << model->getInputSize() << " and output size: " << model->getOutputSize() << endl;
    model->addLayer(new ReLU(5, 5, 2)); // Example layer with input size 784, output size 10, and batch size 32
    model->addLayer(new ReLU(5, 5, 2)); // Example layer with input size 784, output size 10, and batch size 32
    model->addLayer(new Softmax(5, 2)); // Example layer with input size 784, output size 10, and batch size 32
    


    // Call the function
    float* arr = createAndReturnRandomArray(size);
    cout << "Image Pointer: " << arr << endl;



    // Add layers to the model
    float* output = model->forward(arr);



    // Print the array
    cout << "";
    for (int i = 0; i < size; ++i) {
        cout << arr[i] << " ";
    }
    cout << endl;

    cout << endl;
    cout << "";
    for (int i = 0; i < model->getOutputSize(); ++i) {
        cout << output[i] << " ";
    }
    cout << endl;
    

    return 0;
}


