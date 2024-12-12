#include "ls_qr_householder.c"
#include "../../dataset/sns/mk12-b2_fixed.h"
#include "dsgemm.c"


// (float *c_matrix, const float *a_matrix,
//             const int *b_matrix_indptr,
//             const int *b_matrix_indices, const float *b_matrix_data,
//             const unsigned int m_dim, const unsigned int n_dim,
//             const unsigned int k_dim)
int main() {
    float* x_ls;
    float* sketched_a_matrix = (float*)calloc(D_DIM*N_DIM, sizeof(float));
    float* sketched_b_vec = (float*)calloc(D_DIM, sizeof(float));
    dsgemm_csc(sketched_a_matrix, sketching_matrix, a_matrix_indptr,a_matrix_indices, a_matrix_data, D_DIM, N_DIM, M_DIM);
    gemm(sketched_b_vec, sketching_matrix, b_vec, D_DIM, 1, M_DIM);
    x_sketched = householderQRLS(sketched_a_matrix, sketched_b_vec, D_DIM, N_DIM);
    FILE *fptr;
    fptr = fopen("results/custom1_interval.txt", "w");
    if (fptr == NULL)
    {
        printf("Output file Error!");
    }
    printf("\n");
    printf("\n\n");
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

