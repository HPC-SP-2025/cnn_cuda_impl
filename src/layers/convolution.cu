#include "../../include/cnn_library/layers/convolution.h"
#include <cstring>
#include <iostream>

using namespace std;

float *Convolution::forward_CPU(float *input, float *kernel) {
    int H = input_height;
    int W = input_width;
    int oh = output_height;
    int ow = output_width;

    memset(host_forward_buffer, 0, sizeof(float) * output_size);

    for (int b = 0; b < batch_size; b++) {
        int img_idx = b * input_channels * H * W;
        int out_idx = b * output_channels * oh * ow;
        for (int oc = 0; oc < output_channels; ++oc) {
            for (int oy = 0; oy < oh; ++oy) {
                for (int ox = 0; ox < ow; ++ox) {
                    float sum = host_biases[oc];
                    for (int ic = 0; ic < input_channels; ++ic) {
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int iy = oy + ky - padding;
                                int ix = ox + kx - padding;
                                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                    sum += input[img_idx + (ic * H + iy) * W + ix] *
                                           kernel[((oc * input_channels + ic) * kernel_size + ky) * kernel_size + kx];
                                }
                            }
                        }
                    }
                    host_forward_buffer[out_idx + (oc * oh + oy) * ow + ox] = sum;
                }
            }
        }
    }
    return host_forward_buffer;
}

__global__ void conv_forward_kernel(float *input, float *kernel, float *biases, float *output, int n, int in_channels,
                                    int out_channels, int H, int W, int K, int OH, int OW) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n * out_channels * OH * OW) {
        int ox = idx % OW;
        int oy = (idx / OW) % OH;
        int oc = (idx / (OW * OH)) % out_channels;
        int b = idx / (out_channels * OH * OW);

        float sum = 0.0f;

        for (int ic = 0; ic < in_channels; ic++)
            for (int ky = 0; ky < K; ky++)
                for (int kx = 0; kx < K; kx++) {
                    int iy = oy + ky;
                    int ix = ox + kx;

                    if (iy < H && ix < W) {
                        int input_idx = b * in_channels * H * W + ic * H * W + iy * W + ix;
                        int kernel_idx = oc * in_channels * K * K + ic * K * K + ky * K + kx;
                        sum += input[input_idx] * kernel[kernel_idx];
                    }
                }
        int output_idx = b * out_channels * OH * OW + oc * OH * OW + oy * OW + ox;
        output[output_idx] = sum;
    }
}

float *Convolution::forward_GPU(float *input, float *kernel) {

    int threads = 256;
    int num_blocks = (output_size + threads - 1) / threads;

    conv_forward_kernel<<<num_blocks, threads>>>(input, kernel, device_biases, device_forward_buffer, batch_size,
                                                 input_channels, output_channels, input_height, input_width,
                                                 kernel_size, output_height, output_width);

    return device_backward_buffer;
}

float *Convolution::backward_CPU(float *grad_input) {
    int H = input_height;
    int W = input_width;
    int oh = output_height;
    int ow = output_width;

    memset(host_backward_buffer, 0, sizeof(float) * input_size);
    memset(host_grad_weights, 0, sizeof(float) * input_channels * output_channels * kernel_size * kernel_size);
    memset(host_grad_biases, 0, sizeof(float) * output_channels);

    for (int b = 0; b < batch_size; b++) {
        int out_start = b * output_channels * ow * oh;
        int img_start = b * input_channels * H * W;
        for (int oc = 0; oc < output_channels; ++oc) {
            for (int oy = 0; oy < oh; ++oy) {
                for (int ox = 0; ox < ow; ++ox) {
                    int out_index = out_start + (oc * oh + oy) * ow + ox;
                    float grad = grad_input[out_index];

                    host_grad_biases[oc] += grad;

                    for (int ic = 0; ic < input_channels; ++ic) {
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int iy = oy + ky - padding;
                                int ix = ox + kx - padding;
                                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                    int weight_idx = ((oc * input_channels + ic) * kernel_size + ky) * kernel_size + kx;
                                    host_grad_weights[weight_idx] +=
                                        cached_input[img_start + (ic * H + iy) * W + ix] * grad;

                                    host_backward_buffer[img_start + (ic * H + iy) * W + ix] +=
                                        host_weights[weight_idx] * grad;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return host_backward_buffer;
}

__global__ void conv_backward_kernel(float *grad_output, float *input, float *kernel, float *grad_input, int n,
                                     int in_channels, int out_channels, int H, int W, int K, int OH, int OW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * in_channels * H * W;

    if (idx < total) {
        int iw = idx % W;
        int ih = (idx / W) % H;
        int ic = (idx / (W * H)) % in_channels;
        int b = idx / (in_channels * H * W);

        float sum = 0.0f;

        for (int oc = 0; oc < out_channels; ++oc) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int oh = ih - kh;
                    int ow = iw - kw;

                    if (oh >= 0 && oh < OH && ow >= 0 && ow < OW) {
                        int grad_output_idx = b * out_channels * OH * OW + oc * OH * OW + oh * OW + ow;
                        int kernel_idx = oc * in_channels * K * K + ic * K * K + kh * K + kw;
                        sum += grad_output[grad_output_idx] * kernel[kernel_idx];
                    }
                }
            }
        }

        int grad_input_idx = b * in_channels * H * W + ic * H * W + ih * W + iw;
        grad_input[grad_input_idx] = sum;
    }
}

float *Convolution::backward_GPU(float *grad_input) {
    int threads = 256;
    int num_blocks = (input_size + threads - 1) / threads;

    conv_backward_kernel<<<num_blocks, threads>>>(device_backward_buffer, cached_input, device_weights, grad_input,
                                                  batch_size, input_channels, output_channels, input_height,
                                                  input_width, kernel_size, output_height, output_width);

    return device_backward_buffer;
}

Convolution::Convolution(int batch_size, int in_channels, int height, int width, int out_channels, int kernel_size,
                         int padding = 0) {
    this->layer_name = "Conv2D";

    // image
    this->batch_size = batch_size;
    this->input_channels = in_channels;
    this->input_height = height;
    this->input_width = width;

    // kernel
    this->output_channels = out_channels;
    this->kernel_size = kernel_size;
    // this->stride = stride;
    this->padding = padding;

    this->output_height = height + 2 * padding - kernel_size + 1;
    this->output_width = width + 2 * padding - kernel_size + 1;

    initializeWeights();
    initializeBiases();

    this->input_size = batch_size * input_channels * input_height * input_width;
    this->output_size = batch_size * out_channels * output_height * output_width;

    host_forward_buffer = new float[output_size];
    host_backward_buffer = new float[input_size];

    host_grad_weights = new float[in_channels * out_channels * kernel_size * kernel_size];
    host_grad_biases = new float[out_channels];
}

Convolution::~Convolution() {
    delete[] host_weights;
    delete[] host_biases;
    delete[] host_forward_buffer;
    delete[] host_backward_buffer;
    delete[] host_grad_weights;
    delete[] host_grad_biases;

    if (device) {
        cudaFree(device_weights);
        cudaFree(device_biases);
        cudaFree(device_forward_buffer);
        cudaFree(device_backward_buffer);
        cudaFree(device_grad_weights);
        cudaFree(device_grad_biases);
    }
}

void Convolution::setDevice(int dev) {
    device = dev;
    if (device) {
        size_t kernel_space = input_channels * output_channels * kernel_size * kernel_size;
        cudaMalloc(&device_weights, sizeof(float) * kernel_space);
        cudaMalloc(&device_biases, sizeof(float) * output_channels);
        cudaMemcpy(device_weights, host_weights, sizeof(float) * kernel_space, cudaMemcpyHostToDevice);
        cudaMemcpy(device_biases, host_biases, sizeof(float) * output_channels, cudaMemcpyHostToDevice);

        cudaMalloc(&device_forward_buffer, sizeof(float) * output_size);
        cudaMalloc(&device_backward_buffer, sizeof(float) * input_size);

        cudaMalloc(&device_grad_weights, sizeof(float) * kernel_space);
        cudaMalloc(&device_grad_biases, sizeof(float) * output_channels);
    }
}

void Convolution::initializeWeights() {
    size_t num_elems = input_channels * output_channels * kernel_size * kernel_size;
    host_weights = new float[num_elems];
    for (size_t i = 0; i < num_elems; ++i)
        host_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
}

void Convolution::initializeBiases() {
    host_biases = new float[output_channels];
    memset(host_biases, 0, sizeof(float) * output_channels);
}

float *Convolution::forward(float *input) {
    cached_input = input;
    if (device)
        return forward_GPU(input, device_weights);

    else
        return forward_CPU(input, host_weights);
}

float *Convolution::backward(float *grad_input) {
    if (device)
        return backward_GPU(grad_input);

    else
        return backward_CPU(grad_input);
}

void Convolution::updateParameters(float learning_rate) {
    if (device) {
    } else {

        for (size_t i = 0; i < output_channels * input_channels * kernel_size * kernel_size; i++) {
            host_weights[i] -= learning_rate * host_grad_weights[i];
        }

        for (size_t i = 0; i < output_channels; i++) {
            host_biases[i] -= learning_rate * host_grad_biases[i];
        }
    }
}
