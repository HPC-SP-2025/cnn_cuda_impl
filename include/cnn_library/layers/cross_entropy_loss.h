#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H

#include "loss.h"

class Cross_Entropy_Loss : public Loss {
  public:
    // Constructor
    Cross_Entropy_Loss(size_t num_classes, size_t batch_size);

    // Destructor
    ~Cross_Entropy_Loss();

    // Forward pass override
    float *forward(float *pred) override;

    // Backward pass override
    float *backward(float *pred) override;

    // Set the device ID for the layer
    void setDevice(int device) override;

    virtual void setTarget(float *target);

  private:
    float forward_CPU(const float *pred, float *target);
    void backward_CPU(float *grad_output, float *pred, float *target);
    // float forward_GPU(const float *pred, float *target);

  protected:
    float *target;
    float *d_loss;
};

// __global__ void forward_cel_kernel(const float *pred, float *target, float *loss, int n, int num_classes);
// __global__ void backward_cel_kernel(float *grad_output, float *pred, float *target, int n, int num_classes);

#endif // LOSS_H