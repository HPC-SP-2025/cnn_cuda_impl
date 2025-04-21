#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include "loss.h"

class MSE_Loss : public Loss {
  public:
    // Constructor
    MSE_Loss(size_t batch_size);

    // Destructor
    ~MSE_Loss();

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

  protected:
    float *target;
    float *d_loss;
};

__global__ void forward_mse_kernel(const float *pred, float *target, float *loss, int n, int num_classes);
__global__ void backward_mse_kernel(float *grad_output, float *pred, float *target, int n, int num_classes);

#endif // LOSS_H