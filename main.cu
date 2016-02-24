#include <iostream>
using namespace std;

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
    if (row < M.height and col < M.width) {
        M.elements[row * M.stride + col] = value;
    }
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

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C) {
    // Load A and B to device memory
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

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, (A.height + dimBlock.y - 1) / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix Csub = getSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    float temp = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (A.width + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = getSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = getSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = getElement(Asub, row, col);
        Bs[row][col] = getElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();

        if (BLOCK_SIZE * blockCol + col < B.width and BLOCK_SIZE * blockRow + row < A.height) {
            for (int e = 0; e < BLOCK_SIZE and m * BLOCK_SIZE + e < A.width; ++e) {
                Cvalue += As[row][e] * Bs[e][col];
                temp += Cvalue + 1;
            }
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    setElement(Csub, row, col, col);
    //if (row == 1 and col == 1) setElement(Csub, 1, 1, Csub.stride);
}

int main() {
    const int N = 3, M = 1;
    
    float A[N][M], B[M][N], C[N][N], D[N][N];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            A[i][j] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5);
            B[j][i] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5);
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

    cout << "A: " << A[0][0] << " " << A[1][0] << " " << A[2][0] << endl;
    cout << "B: " << B[0][0] << " " << B[0][1] << " " << B[0][2] << endl;
    
    Matrix Am; Am.elements = A[0]; Am.height = N; Am.stride = Am.width = M;
    Matrix Bm; Bm.elements = B[0]; Bm.height = M; Bm.stride = Bm.width = N;
    Matrix Dm; Dm.elements = D[0]; Dm.height = N; Dm.stride = Dm.width = N;

    MatMul(Am, Bm, Dm);

    const float EPS = 0.00001;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (abs(D[i][j] - C[i][j]) > EPS) {
                std::cout << C[i][j] << " " << D[i][j] << "; " << i << ", " << j << std::endl;
            }
        }
    }

    return 0;
}