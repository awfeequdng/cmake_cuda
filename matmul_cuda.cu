#include "matmul.hh"


// Matrix multiplication kernel called by MatMul()
 __global__ void matmul_kernel(Matrix A, Matrix B, Matrix C) {
    C.element[0] = A.element[0] + B.element[0];
 }

void matmul(Matrix& A, Matrix& B, Matrix& C) {
    Matrix d_A, d_B, d_C;
    auto size = A.height * A.weight * sizeof(double);
    cudaMalloc(&d_A.element, size);
    cudaMemcpy(d_A.element, A.element, size, cudaMemcpyHostToDevice);

    size = B.height * B.weight * sizeof(double);
    cudaMalloc(&d_B.element, size);
    cudaMemcpy(d_B.element, B.element, size, cudaMemcpyHostToDevice);

    matmul_kernel<<<1, 1>>>(d_A, d_B, d_C);

    __syncthreads();

    cudaFree(d_A.element);
    cudaFree(d_B.element);
    cudaFree(d_C.element);
}