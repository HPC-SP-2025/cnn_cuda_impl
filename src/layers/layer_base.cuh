#ifndef LAYER_BASE_CUH
#define LAYER_BASE_CUH
#include <vector>

// Abstract base class for layers in a neural network
class Layer {
public:
    // Constructor
    Layer();
    virtual ~Layer();

    // Pure virtual functions to be implemented by derived classes
    virtual void forward(const float* input, float* output, int batchSize);
    virtual void backward(const float* dOutput, float* dInput, int batchSize);
    virtual void updateWeights(float learningRate);

    // Utility functions
    virtual size_t getInputSize();
    virtual size_t getOutputSize();
};

#endif // LAYER_BASE_CUH