#include "kernel.h"
#include <iostream>

void init_matrix(float *matrix, int width, int height, float value)
{
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            matrix[i * width + j] = value;
        }
    }
}

void print_matrix(float *matrix, int width, int height)
{
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            if(matrix[i * width + j] < 10)
            {
                std::cout << 0 << matrix[i * width + j] << " ";
            }
            else
            {
                std::cout << matrix[i * width + j] << " ";
            }
        }
        std::cout << std::endl;
    }
}

#define N 64
int main()
{
    int k = 5;
    int size = N*N*sizeof(float);
    int filt_size = k*k*sizeof(float);
    float *N_h = (float*)malloc(size);
    float *P_h = (float*)malloc(size);
    float *F_h = (float*)malloc(filt_size);
    init_matrix(N_h, N, N, 1.0);
    init_matrix(P_h, N, N, 0.0);
    init_matrix(F_h, k, k, 1.0);
    float *N_d, *P_d, *F_d;
    cudaMalloc(&N_d, size);
    cudaMalloc(&P_d, size);
    cudaMalloc(&F_d, filt_size);
    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(P_d, P_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F_h, filt_size, cudaMemcpyHostToDevice);

    dim3 grid = {N/32, N/32};
    dim3 block = {32, 32};
    conv2d_kernel<<<grid, block>>>(N_d, F_d, P_d, 2, N, N);

    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);
    print_matrix(P_h, N, N);
}