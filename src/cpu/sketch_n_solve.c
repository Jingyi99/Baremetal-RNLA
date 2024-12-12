#include "ls_qr_householder.c"
#include "../../dataset/sns/sk2.h"
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
    x_ls = householderQRLS(sketched_a_matrix, sketched_b_vec, D_DIM, N_DIM);
    printf("\n");
    printf("\n\n");
    printf("result x:");
    for (int i = 0; i < N_DIM; i++) {
        printf("%lf ", x_ls[i]);
    }
    printf("\n");
    free(x_ls);
    free(sketched_a_matrix);
    free(sketched_b_vec);
}

