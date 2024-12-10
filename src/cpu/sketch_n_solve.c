#include "ls_qr_householder.c"
#include "../../dataset/ls/test_ls3.h"

int main() {
    double* x_ls;
    x_ls = householderQRLS(a_matrix, b_vec, M_DIM, N_DIM);
    printf("\n");
    printf("ref x:");
    for (int i = 0; i < N_DIM; i++) {
        printf("%lf ", x_vec[i]);
    }
    printf("\n\n");
    printf("result x:");
    for (int i = 0; i < N_DIM; i++) {
        printf("%lf ", x_ls[i]);
    }
    printf("\n");
    free(x_ls);
}

