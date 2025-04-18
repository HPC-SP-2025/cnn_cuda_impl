#include "include/cnn_library/layers/loss.h"
#include <vector>
#include <stdexcept>

class MSE_Loss : public Loss
{
public:
    MSE_Loss() {}
    ~MSE_Loss() {}

    virtual void forward(const std::vector<float> &input, std::vector<float> &output) = 0;
    virtual void backward(const std::vector<float> &grad_output, std::vector<float> &grad_input) = 0;

private:
    int forward_CPU(const std::vector<float> &output, std::vector<float> &target)
    {
        int n = output.size();
        if (n != target.size())
            throw invalid_argument("Input and target size are not same");

        float loss = 0.0;
        for (auto i = 0; i < n; i++)
        {
            loss += pow(output[i] - target[i], 2);
        }
        loss /= static_cast<float>(n);
        return loss;
    }

    // TODO: use reduction to avoid atomic adds
    __global__ void forward_kernel(vector<float> &output, vector<float> &target, float *loss, int n)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n)
        {
            float diff = output[idx] - target[idx];
            atomicAdd(loss, diff * diff / n);
        }
    }

    void backward_CPU(vector<float> &grad_output, vector<float> &output, vector<float> &target)
    {
        int n = output.size();
        grad_output.resize(n);
        for (auto i = 0; i < grad_output.size(); i++)
        {
            grad_output[i] = 2.0 * (output[i] - target[i]) / n;
        }
    }

    __global__ void backward_kernel(vector<float> &grad_output, vector<float> &output, vector<float> &target, int n)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n)
        {
            grad_output[idx] = 2.0 * (output[idx] - target[idx]) / n;
        }
    }
};