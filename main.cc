#include <iostream>
#include <cmath>
#include <chrono>
#include "matmul.hh"

const int M = 10000;
const int N = 10000;
const int K = 10000;


double A[M][N] = {0.0};
double B[N][K] = {0.0};
double C[M][K] = {0.0};

int main() {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = sin(i * j);
        }
        for (int j = 0; j < K; j++) {
            C[i][j] = 0.5;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            B[i][j] = cos(i * j);
        }
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    Matrix a, b, c;
    a.height = c.height = M;
    a.weight = b.height = N;
    b.weight = c.weight = K;

    matmul(a, b, c);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "matmul spend time: " << double(duration.count()) / 1000000 << " s" << std::endl;
    return 0;
}