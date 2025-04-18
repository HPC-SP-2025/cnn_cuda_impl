#ifndef LOSS_H
#define LOSS_H

#include <vector>

#include "cnn_library/layers/base_layer.h"

class Loss : public Layer {
  public:
    // Constructor
    Loss();

    // Destructor
    ~Loss();

    // Forward pass override
    void forward(const std::vector<float> &input, std::vector<float> &output) override;

    // Backward pass override
    void backward(const std::vector<float> &grad_output, std::vector<float> &grad_input) override;

    // Set the device ID for the layer
    void setDevice(int device) override;

    // Get input size
    size_t getInputSize() override;

    // Get output size
    size_t getOutputSize() override;

  private:
};

#endif // LOSS_H