#include "matrix.hpp"
#include <chrono>
#include <iostream>
#include <vector>

using ms = std::chrono::duration<double, std::milli>;

double time_naive(size_t n, int runs) {
    Matrix<float> a = Matrix<float>::random(n, n);
    Matrix<float> b = Matrix<float>::random(n, n);
    double total = 0;
    for (int i = 0; i < runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto c = a.matmul_naive(b);
        auto end = std::chrono::high_resolution_clock::now();
        total += ms(end - start).count();
    }
    return total / runs;
}

double time_blas(size_t n, int runs) {
    Matrix<float> a = Matrix<float>::random(n, n);
    Matrix<float> b = Matrix<float>::random(n, n);
    double total = 0;
    for (int i = 0; i < runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto c = a.matmul(b);
        auto end = std::chrono::high_resolution_clock::now();
        total += ms(end - start).count();
    }
    return total / runs;
}

#ifdef USE_METAL
double time_metal(size_t n, int runs) {
    Matrix<float> a = Matrix<float>::random(n, n);
    Matrix<float> b = Matrix<float>::random(n, n);
    double total = 0;
    for (int i = 0; i < runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto c = a.matmul_metal(b);
        auto end = std::chrono::high_resolution_clock::now();
        total += ms(end - start).count();
    }
    return total / runs;
}
#endif

int main() {
    std::vector<size_t> sizes = {64, 128, 256, 512, 1024, 2048};
    int runs = 5;

#ifdef USE_METAL
    std::cout << "size    naive(ms)    blas(ms)    metal(ms)    blas_speedup    metal_speedup\n";
    std::cout << "----------------------------------------------------------------------------\n";
    for (size_t n : sizes) {
        double naive = time_naive(n, runs);
        double blas  = time_blas(n, runs);
        double metal = time_metal(n, runs);
        std::cout << n
                  << "    " << naive
                  << "    " << blas
                  << "    " << metal
                  << "    " << naive / blas  << "x"
                  << "    " << naive / metal << "x\n";
    }
#else
    std::cout << "size    naive(ms)    blas(ms)    speedup\n";
    std::cout << "------------------------------------------\n";
    for (size_t n : sizes) {
        double naive = time_naive(n, runs);
        double blas  = time_blas(n, runs);
        std::cout << n
                  << "    " << naive
                  << "    " << blas
                  << "    " << naive / blas << "x\n";
    }
#endif

    return 0;
}
