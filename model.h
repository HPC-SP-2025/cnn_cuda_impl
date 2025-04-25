#ifndef MNIST_MODEL_H
#define MNIST_MODEL_H

# include "cnn_library/nn/sequential.h"

// Function to create the MNIST model
Sequential* create_mnist_model(unsigned int input_size, unsigned int num_classes, unsigned int batch_size); // Renamed to avoid overloading issues

#endif // MNIST_MODEL_H