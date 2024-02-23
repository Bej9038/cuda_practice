#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H

__global__ void conv2d_kernel(const float *N, const float *F, float *P, int r, int width, int height);

#endif //CUDA_KERNEL_H
