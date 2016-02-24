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

__host__ void Transpose(const Matrix A, Matrix B) {
    assert(A.width == B.height and A.height == B.width);

    
}

__global__ void transposeKernel(Matrix A, Matrix B) {

}

int main() {
    // Код для проверки
    const int N = 64, M = 64;

    static float A[N][M], B[M][N], C[M][N];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            A[i][j] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5);
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            B[j][i] = A[i][j];
        }
    }

    Matrix Am; Am.elements = A[0]; Am.height = N; Am.stride = Am.width = M;
    Matrix Cm; Cm.elements = C[0]; Cm.height = M; Cm.stride = Cm.width = N;

    Transpose(Am, Cm);

    const float EPS = 0.00001;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (abs(C[i][j] - B[i][j]) > EPS) {
                std::cout << B[i][j] << " " << C[i][j] << "; " << i << ", " << j << std::endl;
                return 0;
            }
        }
    }

    return 0;
}