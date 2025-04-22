#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "../layers/base_layer.h"
#include "../layers/loss.h"
#include <vector>

class Sequential {

  private:
    int input_size;
    int output_size;
    int device;
    std::vector<Layer *> layers;

  public:
    // Constructor
    Sequential(int input_size, int output_size);

    // Destructor
    ~Sequential();

    // Add a layer to the model
    void addLayer(Layer *layer);

    // Set the device ID for the layer
    void setDevice(int device);

    // Forward pass
    float* forward(float *input);

    // Backward pass
    float backward(float *prediected, float *ground_truth, Loss *loss);

    // Getters for input and output sizes
    int getInputSize() const;
    int getOutputSize() const;

    // Update Parameters
    void updateParameters(float learning_rate);

    // Get Gradients
    void getGradients(std::vector<float> &gradients);

    // Get Params
    void getParams(std::vector<float> &params);

    void saveModel(const string filename);

    // Save Params
    void summary();

    // Load Model Weights
    void loadModel(const string filename);
};

#endif // SEQUENTIAL_H
