#include <stdio.h>
#include <math.h>
#include <riscv.h>

#include "lib.h"
#include "../dataset/small_sdcsc.h"
#include "../src/cpu/dsgemm.c"
// #include "../src/cpu/gemm.c"


// int test_gemm_cpu() {
//     printf("TEST CPU MATMUL\n");
//     data_t results_data[M_DIM*N_DIM] = {0};
//     int start = read_cycles();
//     // gemm(results_data, a_matrix, b_matrix,
//     //           M_DIM, N_DIM, K_DIM);
//     int cycles = read_cycles() - cycles;
//     printf("Verifying results....\n");
//     int error = verify_matrix(verify_data, results_data, M_DIM, N_DIM);
//     printf("%s (%lu cycles)\n", !error ? "PASS" : "FAIL", cycles);
// }

// int test_dsgemm_cpu() {
//     printf("TEST CPU DENSE SPARSE MATMUL\n");
//     data_t results_data[M_DIM*N_DIM] = {0};
//     int start = read_cycles();
//     // dsgemm_csc(results_data, a_matrix, b_matrix_indptr,
//     //         b_matrix_indices, b_matrix_data,
//     //           M_DIM, N_DIM, K_DIM);
//     int cycles = read_cycles() - cycles;
//     printf("Verifying results....\n");
//     int error = verify_matrix(verify_data, results_data, M_DIM, N_DIM);
//     printf("%s (%lu cycles)\n", !error ? "PASS" : "FAIL", cycles);
// }

int test_sdgemm_cpu() {
    printf("TEST CPU SPARSE DENSE MATMUL\n");
    data_t results_data[M_DIM*N_DIM] = {0};
    int start = read_cycles();
    sdgemm_csc(results_data, a_matrix_indptr, a_matrix_indices, a_matrix_data,
                b_matrix, M_DIM, N_DIM, K_DIM);
    int cycles = read_cycles() - cycles;
    printf("Verifying results....\n");
    int error = verify_matrix(results_data, verify_data, M_DIM, N_DIM);
    printf("%s (%lu cycles)\n", !error ? "PASS" : "FAIL", cycles);
    printf("error code %d\n", error);
}

int main () {
    // test_gemm_cpu();
    test_sdgemm_cpu();
    // return 0;
}
