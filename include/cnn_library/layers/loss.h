#ifndef LOSS_H
#define LOSS_H

#include "base_layer.h"

class Loss : public Layer {
  public:
    // Constructor
    Loss() {};

    // Destructor
    ~Loss() override = default;

    // Forward pass override
    virtual float *forward(float *pred) override = 0;

    // Backward pass override
    virtual float *backward(float *pred) override = 0;

    // Set the device ID for the layer
    virtual void setDevice(int device) override = 0;

    virtual void setTarget(float *target) = 0;

    size_t getInputSize() { return input_size; };

    size_t getOutputSize() { return 1; };

    size_t numParams() override { return 0; };

    string getLayerName() override { return this->layer_name; }

    int getDevice() override { return this->device; }

  private:
  protected:
    float *target;
};

#endif // LOSS_H