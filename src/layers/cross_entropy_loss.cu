#include "include/cnn_library/layers/loss.h"
#include <stdexcept>
#include <vector>

class CrossEntropy_Loss : public Loss {
  public:
    CrossEntropy_Loss() {}
    ~CrossEntropy_Loss() {}

    virtual void forward(const std::vector<float> &input, std::vector<float> &output) = 0;
    virtual void backward(const std::vector<float> &grad_output, std::vector<float> &grad_input) = 0;

  private:
    const double EPSILON = 1e-9;

    /**
     * output: Flattened 1D vector with class probabilites
     * target: class label
     */
    int forwardCPU(const std::vector<float> &output, std::vector<float> &target) {
        int n = target.size();
        // if (n != target.size())
        //     throw invalid_argument("Input and target size are not same");

        int num_classes = output.size() / n;

        float loss = 0.0;
        for (auto i = 0; i < n; i++) {
            loss -= log(output[i * num_classes + target[i]] + EPSILON);
        }
        loss /= static_cast<float>(n);
        return loss;
    }

    __global__ void forward_kernel(float *output, float *target, float *loss, int n, int num_classes) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n) {
            float loss_i = log(output[idx * num_classes + (int)target[idx]] + EPSILON);
            atomicAdd(loss, -loss_i);
        }
    }

    void backwardCPU(vector<float> &grad_output, vector<float> &output, vector<float> &target) {
        grad_output.resize(output.size());

        int n = target.size();
        int num_classes = output.size() / n;
        for (auto i = 0; i < grad_output.size(); i++) {
            for (auto j = 0; j < num_classes; j++)
                grad_output[i * num_classes + j] = output[i * num_classes + j] / n;

            grad_output[i * num_classes + target[i]] -= 1.0 / n;
        }
    }

    __global__ void backward_kernel(float *grad_output, float *output, float *target, int n, int num_classes) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n) {
            for (auto j = 0; j < num_classes; j++)
                grad_output[idx * num_classes + j] = output[idx * num_classes + j] / n;

            grad_output[idx * num_classes + (int)target[idx]] -= 1.0 / n;
        }
    }
};