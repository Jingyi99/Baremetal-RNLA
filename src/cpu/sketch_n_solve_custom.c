#include "ls_qr_householder.c"
#include "dsgemm.c"
#if HEADER_FILE == "sk_32_fixed.h"
#include "../../dataset/sns/sk_32_fixed.h"
#elif HEADER_FILE == "sk_32_interval.h"
#include "../../dataset/sns/sk_32_interval.h"
#elif HEADER_FILE == "sk_256_fixed.h"
#include "../../dataset/sns/sk_256_fixed.h"
#elif HEADER_FILE == "sk_256_interval.h"
#include "../../dataset/sns/sk_256_interval.h"
#elif HEADER_FILE == "sk_512_fixed.h"
#include "../../dataset/sns/sk_512_fixed.h"
#elif HEADER_FILE == "sk_512_interval.h"
#include "../../dataset/sns/sk_512_interval.h"
#elif HEADER_FILE == "sk_1024_fixed.h"
#include "../../dataset/sns/sk_1024_fixed.h"
#elif HEADER_FILE == "sk_1024_interval.h"
#include "../../dataset/sns/sk_1024_interval.h"
#elif HEADER_FILE == "sk_4096_fixed.h"
#include "../../dataset/sns/sk_4096_fixed.h"
#elif HEADER_FILE == "sk_4096_interval.h"
#include "../../dataset/sns/sk_4096_interval.h"
#elif HEADER_FILE == "sk_8192_fixed.h"
#include "../../dataset/sns/sk_8192_fixed.h"
#elif HEADER_FILE == "sk_8192_interval.h"
#include "../../dataset/sns/sk_8192_interval.h"
#else
#error "No valid HEADER_FILE specified"
#endif


int main(int argc, char* argv[]) {
    if (argc != 2) {
        return 1;
    }
    char* matrix_file = argv[1];
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
    printf("sketched result x:");
    fprintf(fptr, "sketched result x:");
    for (int i = 0; i < N_DIM; i++) {
        printf("%lf ", x_ls[i]);
        fprintf(fptr, "%lf ", x_ls[i]);
    }
    printf("\n");
    fprintf(fptr, "\n");

    FILE *expected_fptr = fopen(expected_x_file, "r");
    if (expected_fptr == NULL) {
        printf("Expected x file Error!");
        fclose(fptr);
        return 1;
    }

    printf("expected x:");
    fprintf(fptr, "expected x:");
    for (int i = 0; i < N_DIM; i++) {
        fscanf(expected_fptr, "%lf", &x_vec[i]);
        printf("%lf ", x_vec[i]);
        fprintf(fptr, "%lf ", x_vec[i]);
    }
    printf("\n");
    fprintf(fptr, "\n");

    fclose(fptr);
    fclose(expected_fptr);
    free(x_ls);
    free(sketched_a_matrix);
    free(sketched_b_vec);

    return 0;
}

