#include <stdio.h>
#include <math.h>
#include <riscv.h>

#include "lib.h"
// #include "../dataset/small_gemm.h"
#include "../dataset/small_csc.h"
// #include "../src/rvv/gemm.c"
#include "../src/rvv/dsgemm.c"

// int test_gemm_rvv() {
//     printf("TEST RVV DENSE MATMUL\n");
//     data_t results_data[M_DIM*N_DIM] = {0};
//     int start = read_cycles();
//     gemm_rvv(results_data, a_matrix, b_matrix, M_DIM, N_DIM, K_DIM);
//     int cycles = read_cycles() - start;
//     printf("Verifying results....\n");
//     int error = verify_matrix(verify_data, results_data, M_DIM, N_DIM);
//     printf("%s (%lu cycles)\n", !error ? "PASS" : "FAIL", cycles);
// }

int test_dsgemm_rvv() {
    printf("TEST RVV DENSE SPARSE CSC MATMUL\n");
    data_t results_data[M_DIM*N_DIM] = {0};
    int start = read_cycles();
    dsgemm_csc_rvv(results_data, a_matrix,
                b_matrix_indptr,
                b_matrix_indices, b_matrix_data,
                M_DIM, N_DIM, K_DIM);
    int cycles = read_cycles() - start;
    printf("Verifying results....\n");
    // for(int i = 0; i < 5; i++) {
    //     for(int j = 0; j < 5; j++) {
    //         printf("%d ", (int)results_data[i*5 + j]);
    //     }
    //     printf("\n");
    // }
    // for(int i = 0; i < 5; i++) {
    //     for(int j = 0; j < 5; j++) {
    //         printf("%d ", (int)verify_data[i*5 + j]);
    //     }
    //     printf("\n");
    // }
    
    int error = verify_matrix(results_data, verify_data, M_DIM, N_DIM);
    printf("%s (%lu cycles)\n", !error ? "PASS" : "FAIL", cycles);
}

int main() {
    test_dsgemm_rvv();
}