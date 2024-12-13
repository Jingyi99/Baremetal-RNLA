// #include "../../test/lib.h"
// #include "../riscv.h"
// #include "../../src/cpu/ls_qr_householder.c"
// #include "../../src/cpu/dsgemm.c"
// #include "../../dataset/test_mats/sk_32_fixed.h"


#include "lib.h"
#include "riscv.h"
// #include "../../src/cpu/ls_qr_householder.c"
// #include "../../src/cpu/dsgemm.c"
#include "sk_32_fixed.h"
#include <stdlib.h>
#include <stdio.h>
#include "dsgemm.h"


extern float* householderQRLS(float* A, float* b, int m, int n);


int main() {
    printf("PREGENERATED SKETCH & SOLVE!\n");
    float* x_ls;
    float* sketched_a_matrix = (float*)calloc(D_DIM*N_DIM, sizeof(float));
    float* sketched_b_vec = (float*)calloc(D_DIM, sizeof(float));
    uint64_t sketching_start = read_cycles();
    dsgemm_csc(sketched_a_matrix, sketching_matrix, a_matrix_indptr,a_matrix_indices, a_matrix_data, D_DIM, N_DIM, M_DIM);
    gemm(sketched_b_vec, sketching_matrix, b_vec, D_DIM, 1, M_DIM);
    int sketching_cycles = read_cycles() - sketching_start;
    printf("Matrix dim: %d by %d, sketching took %lu cycles\n", M_DIM, N_DIM, sketching_cycles);

    uint64_t solver_start = read_cycles();
    x_ls = householderQRLS(sketched_a_matrix, sketched_b_vec, D_DIM, N_DIM);
    uint64_t solving_cycles = read_cycles() - solver_start;

    printf("Matrix dim: %d by %d, sketching took %lu cycles, solving took %lu cycles\n", M_DIM, N_DIM, sketching_cycles, solving_cycles);

    return 0;
}
