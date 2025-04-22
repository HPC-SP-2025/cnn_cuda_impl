#include <iostream>
#include "../include/cnn_library/nn/sequential.h"
#include "../include/cnn_library/layers/relu.h"
#include "../include/cnn_library/layers/softmax.h"
#include "../include/cnn_library/layers/cross_entropy_loss.h"


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

    int size = 5;    


    // Create a Sequential model
    Sequential* model = new Sequential(size, size); // Example input size of 784 (28x28 image flattened) and output size of 10 (number of classes)
    cout << "Model created with input size: " << model->getInputSize() << " and output size: " << model->getOutputSize() << endl;
    model->addLayer(new ReLU(5, 5, 1)); // Example layer with input size 784, output size 10, and batch size 32
    model->addLayer(new ReLU(5, 5, 1)); // Example layer with input size 784, output size 10, and batch size 32
    model->addLayer(new Softmax(5, 1)); // Example layer with input size 784, output size 10, and batch size 32
    model->summary();
    Cross_Entropy_Loss* loss_fn = new Cross_Entropy_Loss(5, 1); // Example layer with input size 784, output size 10, and batch size 32
    //loss_fn->setTarget(new float[1]{4}); // Set the target for the loss function

    
    
    
    for(int j = 0; j < 10; j++)
    {
    // Call the function
    float* arr = createAndReturnRandomArray(size);
    int targetIndex = 0;
    for (int i = 1; i < size; ++i) {
        if (arr[i] > arr[targetIndex]) {
            targetIndex = i;
        }
    }
    loss_fn->setTarget(new float[1]{static_cast<float>(targetIndex)}); // Set the target for the loss function
    // float* arr = new float[size]{-1.0, -.05, 0, 0, 0.8}; // Example input array
    cout << "Image Pointer: " << arr << endl;

    // Add layers to the model
    float* output = model->forward(arr);
    float* loss_value = loss_fn->forward(output);
    cout << "Loss Value: " << *loss_value << endl;
    

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

}
    

    return 0;
}


