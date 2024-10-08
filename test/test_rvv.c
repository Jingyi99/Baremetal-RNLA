#include <stdio.h>
#include <math.h>
#include <riscv.h>

#include "../dataset/small_gemm.h"
#include "../src/rvv/matmul.c"
#include "lib.h"

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