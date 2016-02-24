#include <iostream>
#include <stdio.h>
#include <assert.h>

struct Matrix {
    int width;
    int height;
    int stride;
    float* elements;
};

#define BLOCK_SIZE 8

__global__ void TransposeKernel(Matrix A, Matrix B) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int col = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int gridWidth = gridDim.x * BLOCK_SIZE;

    for (int j = 0; j < BLOCK_SIZE; j += BLOCK_ROWS)
        tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

    __syncthreads();

    x = blockIdx.y * BLOCK_SIZE + threadIdx.x;  // transpose block offset
    y = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    for (int j = 0; j < BLOCK_SIZE; j += BLOCK_ROWS)
        odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

__host__ void Transpose(const Matrix A, Matrix B) {
    assert(A.width == B.height and A.height == B.width);

    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    cudaMalloc(&d_B.elements, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, (A.height + dimBlock.y - 1) / dimBlock.y);
    TransposeKernel <<<dimGrid, dimBlock>>> (d_A, d_B);

    cudaMemcpy(B.elements, d_B.elements, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
}

int main() {
    // Код для проверки
    const int N = 64, M = 64;

    static float A[N][M], B[M][N];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            A[i][j] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5);
        }
    }

    Matrix Am; Am.elements = A[0]; Am.height = N; Am.stride = Am.width = M;
    Matrix Bm; Bm.elements = B[0]; Bm.height = M; Bm.stride = Bm.width = N;

    Transpose(Am, Bm);

    const float EPS = 0.00001;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (abs(A[j][i] - B[i][j]) > EPS) {
                std::cout << A[j][i] << " " << B[i][j] << "; " << i << ", " << j << std::endl;
                return 0;
            }
        }
    }

    return 0;
}