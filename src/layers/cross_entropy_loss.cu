#include "include/cnn_library/layers/loss.h"
#include <stdexcept>
#include <vector>

class CrossEntropy_Loss : public Loss {
  public:
    CrossEntropy_Loss(size_t num_classes, size_t batch_size) {
        this->layer_name = "Cross_Entropy_Loss";
        this->batch_size = batch_size;
        this->input_size = num_classes;

        this->host_forward_buffer = new float[1];
        this->host_backward_buffer = new float[batch_size * num_classes];
    }

    ~CrossEntropy_Loss() {
        delete[] host_forward_buffer;
        delete[] host_backward_buffer;
        if (device) {
            cudaFree(device_backward_buffer);
            cudaFree(d_loss);
        }
    }

    float *forward(const float *pred) {
        float loss;
        if (this->device) {
            cudaMemset(d_loss, 0, sizeof(float));
            forward_kernel<<<1, 256>>>(pred, this->target, d_loss, batch_size, input_size);
            cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

        } else {
            loss = forward_CPU(pred, this->target);
        }
        host_forward_buffer[0] = loss;
        return host_forward_buffer;
    }

    float *backward(float *pred) {
        if (this->device) {
            backward_kernel(device_backward_buffer, pred, this->target, batch_size, input_size);
            return device_backward_buffer;
        } else {
            backward_CPU(host_backward_buffer, pred, this->target);
            return host_backward_buffer;
        }
    }

    void setDevice(int device) override {
        this->device = device;
        if (device) {
            cudaMalloc(&d_loss, sizeof(float));
            cudaMalloc(&device_backward_buffer, sizeof(float) * batch_size * input_size);
        }
    }

    void setTarget(float *target) { this->target = target; }

    /**
     * output: Flattened 1D vector with class probabilites
     * target: class label
     */
    float forward_CPU(const float *pred, float *target) {
        int n = batch_size;
        int num_classes = input_size;

        float loss = 0.0;
        for (auto i = 0; i < n; i++) {
            loss -= log(pred[i * num_classes + (int)target[i]] + EPSILON);
        }
        loss /= static_cast<float>(n);
        return loss;
    }

    __global__ void forward_kernel(const float *pred, float *target, float *loss, int n, int num_classes) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n) {
            float loss_i = log(pred[idx * num_classes + (int)target[idx]] + EPSILON);
            atomicAdd(loss, -loss_i);
        }
    }

    void backward_CPU(float *grad_output, float *pred, float *target) {

        int n = batch_size;
        int num_classes = input_size;
        for (auto i = 0; i < n; i++) {
            for (auto j = 0; j < num_classes; j++)
                grad_output[i * num_classes + j] = pred[i * num_classes + j] / n;

            grad_output[i * num_classes + (int)target[i]] -= 1.0 / n;
        }
    }

    __global__ void backward_kernel(float *grad_output, float *pred, float *target, int n, int num_classes) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n) {
            for (auto j = 0; j < num_classes; j++)
                grad_output[idx * num_classes + j] = pred[idx * num_classes + j] / n;

            grad_output[idx * num_classes + (int)target[idx]] -= 1.0 / n;
        }
    }

  private:
    float *d_loss;

    const double EPSILON = 1e-9;
};