#include "ls_qr_householder.c"
#include "../../dataset/ls/small_ls.h"

int main() {
    float* x_ls;
    x_ls = householderQRLS(a_matrix, b_vec, M_DIM, N_DIM);
    printf("ref x:");
    for (int i = 0; i < N_DIM; i++) {
        printf("%f ", x_vec[i]);
    }
    printf("\n\n");
    printf("result x:");
    for (int i = 0; i < N_DIM; i++) {
        printf("%f ", x_ls[i]);
    }
    printf("\n");
    free(x_ls);
}

