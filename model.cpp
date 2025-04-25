#include <iostream>
#include <string>
#include <vector>
#include "model.h"

// Include modules to build the model
#include "include/cnn_library/layers/relu.h"
#include "include/cnn_library/layers/linear.h"
#include "include/cnn_library/layers/softmax.h"
#include "include/cnn_library/nn/sequential.h"


Sequential *create_mnist_model(unsigned int input_size, unsigned int batch_size, unsigned int num_classes) 
{

    // Create a Sequential model
    Sequential *model = new Sequential(input_size, num_classes);

    // Add layers to the model
    model->addLayer(new Linear(input_size, 1024,batch_size));
    model->addLayer(new ReLU(1024, 1024, batch_size)); 
    model->addLayer(new Linear(1024, 1024, batch_size));
    model->addLayer(new ReLU(1024, 1024, batch_size)); 
    model->addLayer(new Linear(1024, 128, batch_size)); 
    model->addLayer(new ReLU(128, 128, 1));
    model->addLayer(new Linear(128, num_classes, batch_size));
    model->addLayer(new Softmax(num_classes, batch_size));

    // Return
    return model;
}
