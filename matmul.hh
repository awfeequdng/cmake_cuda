#pragma once

struct Matrix {
    int weight;
    int height;
    int stride;
    double *element;
};

void matmul(Matrix& A, Matrix& B, Matrix& C);