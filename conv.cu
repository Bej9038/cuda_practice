#include <iostream>
#include <ctime>
#include "kernel.h"

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

#define N 28
#define R 2
#define BLOCK 32
#define IN_DIM 32

__constant__ float F[2*R+1][2*R+1];

__global__ void conv2d_kernel_tiled_const_mem(const float *M, float *P, int r, int width, int height)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int by = blockIdx.y;
    int ty = threadIdx.y;

    int col = bx * ((IN_DIM) - 2*(R)) + tx - R;
    int row = by * ((IN_DIM) - 2*(R)) + ty - R;

    __shared__ float M_s[IN_DIM][IN_DIM];

    if(row >= 0 and row < height and col >= 0 and col < width)
    { M_s[ty][tx] = M[row*width + col]; }
    else
    { M_s[ty][tx] = 0; }

    __syncthreads();

    int tileCol = tx - R;
    int tileRow = ty - R;

    if(row >= 0 and row < height and col >= 0 and col < width)
    {
        if(tileCol >= 0 and tileCol < ((IN_DIM) - 2*(R)) and tileRow >= 0 and tileRow < ((IN_DIM) - 2*(R)) ) {
            float result = 0;
            for (int i = 0; i < 2*R+1; i++) {
                for (int j = 0; j < 2*R+1; j++) {
                    result += F[i][j] * M_s[tileRow + i][tileCol + j];
                }
            }
            P[row * width + col] = result;
        }
    }
}

int main()
{
    int k = 2*R+1;
    int size = N*N*sizeof(float);
    int filter_size = k * k * sizeof(float);
    float *N_h = (float*)malloc(size);
    float *P_h = (float*)malloc(size);
    float *F_h = (float*)malloc(filter_size);

    init_matrix(N_h, N, N, 1.0);
    init_matrix(P_h, N, N, 0.0);
    init_matrix(F_h, k, k, 1.0);

    float *N_d, *P_d, *F_d;
    cudaMalloc(&N_d, size);
    cudaMalloc(&P_d, size);
    cudaMalloc(&F_d, size);

    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(P_d, P_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F_h, filter_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, F_h, filter_size);


    dim3 grid = {1 , 1};
    dim3 block = {BLOCK, BLOCK};

    clock_t start = clock();
//    conv2d_kernel<<<grid, block>>>(N_d, F_d, P_d, R, N, N);
    conv2d_kernel_tiled_const_mem<<<grid, block>>>(N_d, P_d, 2, N, N);
    cudaDeviceSynchronize();

    clock_t end = clock();
    double elapsed_seconds = double(end - start) / CLOCKS_PER_SEC;

    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);
    print_matrix(P_h, N, N);
    std::cout << "Time elapsed: " << elapsed_seconds << "s\n";

//    free(N_h);
//    free(P_h);
//    free(F_h);
//    cudaFree(N_d);
//    cudaFree(P_d);
//    cudaFree(F_d);

}