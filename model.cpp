# include <iostream>
# include <vector>
# include <string>
# include "model.h" // Include the new header file

// Include modules to build the model
# include "cnn_library/layers/convolution.h"
# include "cnn_library/layers/relu.h"
# include "cnn_library/layers/linear.h"
# include "cnn_library/layers/softmax.h"
# include "cnn_library/layers/loss.h"
# include "cnn_library/nn/sequential.h"
# include "cnn_library/layers/flatten.h"

Sequential* create_mnist_model() {
    Sequential* model = new Sequential();

    // Create 

    // Add layers to the model
    model->addLayer(new Convolution(3, 32, 3, 1)); // Input channels: 1, Output channels: 32, Kernel size: 3x3, Stride: 1
    model->addLayer(new ReLU());
    model->addLayer(new Linear(32 * 26 * 26, 128)); // Flattened input size: 32 * 26 * 26, Output size: 128
    model->addLayer(new ReLU());
    model->addLayer(new Linear(128, 64));
    model->addLayer(new Linear(128, 10)); // Fully connected layer to 10 classes
    model->addLayer(new Softmax());

    return model;
}







