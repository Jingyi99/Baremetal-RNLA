// #include <riscv.h>

#include "ls_qr_householder.c"
#include "../../dataset/sns/sk.h"
#include "dsgemm.c"
#include "../../test/lib.h"
#include "../../test/riscv.h"

int main() {
    printf("PREGENERATED SKETCH & SOLVE\n");
    float* x_ls;
    float* sketched_a_matrix = (float*)calloc(D_DIM*N_DIM, sizeof(float));
    float* sketched_b_vec = (float*)calloc(D_DIM, sizeof(float));
    size_t sketching_start = read_cycles();
    dsgemm_csc(sketched_a_matrix, sketching_matrix, a_matrix_indptr,a_matrix_indices, a_matrix_data, D_DIM, N_DIM, M_DIM);
    gemm(sketched_b_vec, sketching_matrix, b_vec, D_DIM, 1, M_DIM);
    int sketching_cycles = read_cycles() - sketching_start;

    int solver_start = read_cycles();
    x_ls = householderQRLS(sketched_a_matrix, sketched_b_vec, D_DIM, N_DIM);
    int solving_cycles = read_cycles() - solver_start;

    printf("Matrix dim: %d by %d, sketching took %lu cycles, solving took %lu cycles\n", D_DIM, N_DIM, sketching_cycles, solving_cycles);
    FILE *fptr;
    fptr = fopen("results/custom1_interval.txt", "w");
    if (fptr == NULL)
    {
        printf("Output file Error!");
    }
    printf("\n");
    // printf("qrls result x:");
    // for (int i = 0; i < N_DIM; i++) {
    //     printf("%lf ", x_ls[i]);
    // }
    printf("sketched result x:");
    fprintf(fptr, "sketched result x:");
    for (int i = 0; i < N_DIM; i++) {
        printf("%lf ", x_ls[i]);
        fprintf(fptr, "%lf ", x_ls[i]);
    }
    printf("\n");
    fprintf(fptr, "\n");
    // printf("expected x:");
    // fprintf(fptr, "expected x:");
    // for (int i = 0; i < N_DIM; i++) {
    //     printf("%lf ", x_vec[i]);
    //     fprintf(fptr, "%lf ", x_vec[i]);
    // }
    printf("\n");
    fprintf(fptr, "\n");
    fclose(fptr);
    free(x_ls);
    free(sketched_a_matrix);
    free(sketched_b_vec);
}

