#include "include/cnn_library/layers/loss.h"
#include <stdexcept>
#include <vector>

class MSE_Loss : public Loss {
  public:
    MSE_Loss() {}
    ~MSE_Loss() {
        if (device)
            cudaFree(d_loss);
    }

    void forward(const float *pred, float *output) {
        float loss;
        if (this->device == 0) {
            loss = forward_CPU(pred, this->target);
        } else {
            cudaMemset(d_loss, 0, sizeof(float));
            forward_kernel<<<1, 256>>>(pred, this->target, d_loss, batch_size);
            cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
        }
        output[0] = loss;
    }

    void backward(float *pred, float *grad_output) {
        if (this->device == 0) {

            backward_CPU(grad_output, pred, target);
        } else {
            backward_kernel(grad_output, pred, target, batch_size);
        }
    }

    void setDevice(int device) override {
        this->device = device;
        if (device)
            cudaMalloc(&d_loss, sizeof(float));
    }
    void setTarget(float *target) { this->target = target; }

    int forward_CPU(const float *pred, float *target) {
        int n = batch_size;

        float loss = 0.0;
        for (auto i = 0; i < n; i++) {
            loss += pow(pred[i] - target[i], 2);
        }
        loss /= static_cast<float>(n);
        return loss;
    }

    // TODO: use reduction to avoid atomic adds
    __global__ void forward_kernel(const float *pred, float *target, float *loss, int n) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n) {
            float diff = pred[idx] - target[idx];
            atomicAdd(loss, diff * diff / n);
        }
    }

    void backward_CPU(float *grad_output, float *pred, float *target) {
        int n = batch_size;

        for (auto i = 0; i < n; i++) {
            grad_output[i] = 2.0 * (pred[i] - target[i]) / n;
        }
    }

    __global__ void backward_kernel(float *grad_output, float *pred, float *target, int n) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n) {
            grad_output[idx] = 2.0 * (pred[idx] - target[idx]) / n;
        }
    }

  private:
    float *target;
    float *d_loss;
};