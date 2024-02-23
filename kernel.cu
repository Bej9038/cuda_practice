#include "kernel.h"

__global__ void conv2d_kernel(const float *N, const float *F, float *P, int r, int width, int height)
{
    int bx = blockIdx.x;
    int bd = blockDim.x;
    int tx = threadIdx.x;
    int by = blockIdx.y;
    int ty = threadIdx.y;

    int col = bx * bd + tx;
    int row = by * bd + ty;

    float result = 0;
    for(int i = 0; i < 2*r+1; i++)
    {
        for(int j = 0; j < 2*r+1; j++)
        {
            int in_row = row - r + i;
            int in_col = col - r + j;
            if(in_row >= 0 and in_row < height and in_col >= 0 and in_col < width)
            {
                result += F[i * (2*r+1) + j] * N[in_row * width + in_col];
            }
        }
    }
    P[row * width + col] = result;
}