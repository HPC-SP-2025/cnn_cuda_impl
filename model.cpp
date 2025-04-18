# include <iostream>
# include <vector>
# include <string>
# include "model.h" // Include the new header file

// Include modules to build the model
# include "include/cnn_library/layers/convolution.h"
# include "include/cnn_library/layers/relu.h"
# include "inlclude/cnn_library/layers/linear.h"
# include "include/cnn_library/layers/softmax.h"
# include "include/cnn_library/layers/loss.h"
# include "include/cnn_library/nn/sequential.h"
# include "include/cnn_library/layers/flatten.h"

Sequential* create_mnist_model(unsigned int input_size, unsigned int batch_size, unsigned int num_classes) 
{
    // Create a Sequential model
    // Sequential* model = new Sequential(input_size, num_classes);
    // Sequential* model = new Sequential();
    // Create a Sequential model

{
    Sequential* model = new Sequential();

    // Create 

    // Add layers to the model
    model->addLayer(new Linear(input_size, 1024, batch_size)); // Input channels: 1, Output channels: 32, Kernel size: 3x3, Stride: 1
    model->addLayer(new ReLU());
    model->addLayer(new Linear(1024, 1024, batch_size)); // Flattened input size: 32 * 26 * 26, Output size: 128
    model->addLayer(new ReLU());
    model->addLayer(new Linear(1024, 128, batch_size)); // Fully connected layer to 128 neurons
    model->addLayer(new ReLU());
    model->addLayer(new Linear(128, num_classes, batch_size)); // Fully connected layer to 10 classes
    model->addLayer(new Softmax());
    

    return model;
}







