#ifndef LAYER_BASE_CUH
#define LAYER_BASE_CUH
#include <vector>

// Abstract base class for layers in a neural network
class LayerBase {
public:
    virtual ~LayerBase() {}

    // Pure virtual functions to be implemented by derived classes
    virtual void forward(const float* input, float* output, int batchSize) = 0;
    virtual void backward(const float* dOutput, float* dInput, int batchSize) = 0;
    virtual void updateWeights(float learningRate) = 0;

    // Utility functions
    virtual size_t getInputSize() const = 0;
    virtual size_t getOutputSize() const = 0;
};

#endif // LAYER_BASE_CUH