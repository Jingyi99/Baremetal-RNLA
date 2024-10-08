#include <stdio.h>
#include <math.h>
#include <riscv.h>

#include "lib.h"
#include "../dataset/small_csc.h"
#include "../src/cpu/dsgemm.c"
#include "../src/cpu/gemm.c"

int test_gemm_cpu() {
    printf("TEST CPU MATMUL\n");
    data_t results_data[M_DIM*N_DIM] = {0};
    int start = read_cycles();
    // gemm(results_data, a_matrix, b_matrix,
    //           M_DIM, N_DIM, K_DIM);
    int cycles = read_cycles() - cycles;
    printf("Verifying results....\n");
    int error = verify_matrix(verify_data, results_data, M_DIM, N_DIM);
    printf("%s (%lu cycles)\n", !error ? "PASS" : "FAIL", cycles);
}

int test_dsgemm_cpu() {
    printf("TEST CPU DENSE SPARSE MATMUL\n");
    data_t results_data[M_DIM*N_DIM] = {0};
    int start = read_cycles();
    dsgemm(results_data, a_matrix, b_matrix_indptr,
            b_matrix_indices, b_matrix_data,
              M_DIM, N_DIM, K_DIM);
    int cycles = read_cycles() - cycles;
    printf("Verifying results....\n");
    int error = verify_matrix(verify_data, results_data, M_DIM, N_DIM);
    printf("%s (%lu cycles)\n", !error ? "PASS" : "FAIL", cycles);
}

int main () {
    // test_gemm_cpu();
    test_dsgemm_cpu();
    // return 0;
}
