#include <iostream>
#include <stdio.h>
#include <assert.h>

struct Matrix {
    int width;
    int height;
    int stride;
    float* elements;
};

#define BLOCK_SIZE 16

__device__ float getElement(const Matrix M, int row, int col) {
    return (row < M.height and col < M.width ? M.elements[row * M.stride + col] : 0);
}

__device__ void setElement(Matrix M, int row, int col, float value) {
    //printf("row %d, col %d, stride %d\n", row, col, M.stride);
    M.elements[row * M.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix getSubMatrix(Matrix A, int row, int col) {
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

__global__ void ProductKernel(Matrix A, Matrix B, Matrix C) {

    // row/col inside grid
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix Csub = getSubMatrix(C, blockRow, blockCol);

    float Cvalue = 0;

    // row/col inside block
    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (A.width + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {

        Matrix Asub = getSubMatrix(A, blockRow, m);
        Matrix Bsub = getSubMatrix(B, m, blockCol);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = getElement(Asub, row, col);
        Bs[row][col] = getElement(Bsub, row, col);

        __syncthreads();

        if (BLOCK_SIZE * blockCol + col < B.width and BLOCK_SIZE * blockRow + row < A.height) {
            for (int e = 0; e < BLOCK_SIZE and m * BLOCK_SIZE + e < A.width; ++e) {
                Cvalue += As[row][e] * Bs[e][col];
            }
        }

        __syncthreads();
    }

    if (BLOCK_SIZE * blockCol + col < B.width and BLOCK_SIZE * blockRow + row < A.height) {
        setElement(Csub, row, col, Cvalue);
    }
}

__host__ void Product(const Matrix A, const Matrix B, Matrix C) {
    assert(A.width == B.height and C.height == A.height and C.width == B.width);

    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, (A.height + dimBlock.y - 1) / dimBlock.y);
    ProductKernel <<<dimGrid, dimBlock>>> (d_A, d_B, d_C);

    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

int main() {
    // Код для проверки
    const int N = 345, M = 567;
    
    static float A[N][M], B[M][N], C[N][N], D[N][N];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            A[i][j] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5) / 1000;
            B[j][i] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5) / 1000;
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < M; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    Matrix Am; Am.elements = A[0]; Am.height = N; Am.stride = Am.width = M;
    Matrix Bm; Bm.elements = B[0]; Bm.height = M; Bm.stride = Bm.width = N;
    Matrix Dm; Dm.elements = D[0]; Dm.height = N; Dm.stride = Dm.width = N;

    Product(Am, Bm, Dm);

    const float EPS = 0.00001;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (abs(D[i][j] - C[i][j]) > EPS) {
                std::cout << C[i][j] << " " << D[i][j] << "; " << i << ", " << j << std::endl;
                return 0;
            }
        }
    }

    return 0;
}