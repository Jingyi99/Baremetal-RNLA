#include <stdio.h>
#include <math.h>
#include <riscv.h>

#include "../dataset/small_gemm.h"
#include "../src/rvv/matmul.c"
// #include "lib.h"

// returns nonzero if errors
int verify_matrix(float *result, float *gold, size_t m_dim, size_t n_dim) {
    float tolerance = 1e-2;
    for (uint64_t i = 0; i < m_dim; ++i) {
        for (uint64_t j = 0; j < n_dim; ++j) {
            uint64_t idx = i * n_dim + j;
            // printf("i: %u, j: %u, idx: %u, result: %f\n", i, j, idx, tolerance);
            if (fabs(result[idx]-gold[idx]) > tolerance){
                // printf("i: %u, j: %u, idx: %u\n", i, j, idx);
                return (i+j == 0? -1 : idx);
            }
        }
    }
    return 0;
}

int test_matmul_rvv() {
    printf("TEST RVV MATMUL");
    data_t results_data[M_DIM*N_DIM] = {0};
    // start = read_cycles();
    matmul_rvv(results_data, a_matrix, b_matrix,
              M_DIM, N_DIM, K_DIM);
    // cycles = read_cycles() - cycles;
    printf("Verifying results....\n");
    int error = verify_matrix(verify_data, results_data, M_DIM, N_DIM);
    // printf("%s (%lu cycles)\n", error ? "PASS" : "FAIL", cycles);
}
int main () {
    test_matmul_rvv();
    // return 0;
}