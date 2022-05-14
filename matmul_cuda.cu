#include "matmul.hh"

void matmul(Matrix& A, Matrix& B, Matrix& C) {
    Matrix d_A;
    auto size = A.height * A.weight * sizeof(double);
    cudaMalloc(&d_A.element, size);
    cudaMemcpy(d_A.element, A.element, size, cudaMemcpyHostToDevice);

    cudaFree(d_A.element);
}