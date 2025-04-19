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
    void forward(float *pred, float *target) override;

    // Backward pass override
    void backward(float *pred, float *grad_output) override;

    // Set the device ID for the layer
    void setDevice(int device) override;

    // Get input size
    size_t getInputSize() override;

    // Get output size
    size_t getOutputSize() override;

    size_t numParams() override { return 0; };

    string getLayerName() override { return this->layer_name; }

    int getDevice() override { return this->device; }

    virtual void setTarget(float *target);

  private:
  protected:
    float *target;
};

#endif // LOSS_H