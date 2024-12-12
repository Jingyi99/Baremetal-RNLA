#include "ls_qr_householder.c"
#include "dsgemm.c"
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <stdio.h>
#if HEADER_FILE == 1
#include "../../dataset/sns/sk_32_fixed.h"
#elif HEADER_FILE == 2
#elif HEADER_FILE == 2
#include "../../dataset/sns/sk_32_interval.h"
#elif HEADER_FILE == 3
#elif HEADER_FILE == 3
#include "../../dataset/sns/sk_256_fixed.h"
#elif HEADER_FILE == 4
#elif HEADER_FILE == 4
#include "../../dataset/sns/sk_256_interval.h"
#elif HEADER_FILE == 5
#elif HEADER_FILE == 5
#include "../../dataset/sns/sk_512_fixed.h"
#elif HEADER_FILE == 6
#elif HEADER_FILE == 6
#include "../../dataset/sns/sk_512_interval.h"
#elif HEADER_FILE == 7
#elif HEADER_FILE == 7
#include "../../dataset/sns/sk_1024_fixed.h"
#elif HEADER_FILE == 8
#elif HEADER_FILE == 8
#include "../../dataset/sns/sk_1024_interval.h"
#elif HEADER_FILE == 9
#elif HEADER_FILE == 9
#include "../../dataset/sns/sk_4096_fixed.h"
#elif HEADER_FILE == 10
#elif HEADER_FILE == 10
#include "../../dataset/sns/sk_4096_interval.h"
#elif HEADER_FILE == 11
#elif HEADER_FILE == 11
#include "../../dataset/sns/sk_8192_fixed.h"
#elif HEADER_FILE == 12
#elif HEADER_FILE == 12
#include "../../dataset/sns/sk_8192_interval.h"
#endif

// basically transform from csc to csr
void getATranspose(int** a_transpose_indptr, int** a_transpose_indices, float** a_transpose_data) {
    int nnz = a_matrix_indptr[N_DIM];
    float* temp_data = (float*)calloc(nnz, sizeof(float));
    int* temp_indices = (int*)calloc(nnz, sizeof(int));
    int* temp_indptr = (int*)calloc(M_DIM + 1, sizeof(int));
    for (int i = 0; i < N_DIM; i++) {
        for (int j = a_matrix_indptr[i]; j < a_matrix_indptr[i + 1]; j++) {
            int row = a_matrix_indices[j];
            
            int dest = temp_indptr[row + 1]++;
            temp_data[dest] = a_matrix_data[j];
            temp_indices[dest] = i;
        }
    }
    for (int i = 1; i <= M_DIM; i++) {
        temp_indptr[i] += temp_indptr[i - 1];
    }
    *a_transpose_data = temp_data;
    *a_transpose_indices = temp_indices;
    *a_transpose_indptr = temp_indptr;
}

float getL2NormDense(float* matrix, int m, int n) {
    float norm = 0;
    for (int i = 0; i < n; i++) {
        float colSum = 0;
        for (int j = 0; j < m; j++) {
            colSum += matrix[j * n + i] * matrix[j * n + i];
        }
        norm += sqrt(colSum);
    }
    return norm;
}


float getFrobeniusNormSparse(){
    float norm = 0;
    for (int j = 0; j < N_DIM; j++) {
        for (int k = a_matrix_indptr[j]; k < a_matrix_indptr[j+1]; k++) {
            norm += a_matrix_data[k] * a_matrix_data[k];
        }
    }
    return sqrt(norm);
}

// ∥A⊤(Ax − b)∥2 / ∥A∥F ∥Ax − b∥2
float getErrorMetric(float* x_ls) {
    float* Ax = (float*)calloc(M_DIM, sizeof(float));
    sdgemm_csc(Ax, a_matrix_indptr, a_matrix_indices, a_matrix_data, x_ls, M_DIM, 1, N_DIM);
    float* Ax_minus_b = (float*)calloc(M_DIM, sizeof(float));
    for (int i = 0; i < M_DIM; i++) {
        Ax_minus_b[i] = Ax[i] - b_vec[i];
    }
    float* A_transpose_Ax_minus_b = (float*)calloc(N_DIM, sizeof(float));
    int nnz = a_matrix_indptr[N_DIM];
    int* a_transpose_indptr = (int*)calloc(M_DIM+1, sizeof(int));
    int* a_transpose_indices = (int*)calloc(nnz, sizeof(int));
    float* a_transpose_data = (float*)calloc(nnz, sizeof(float));
    getATranspose(&a_transpose_indptr, &a_transpose_indices, &a_transpose_data);
    sdgemm_csc(A_transpose_Ax_minus_b, a_transpose_indptr, a_transpose_indices, a_transpose_data, Ax_minus_b, N_DIM, 1, M_DIM);
    float denominator = getFrobeniusNormSparse() * getL2NormDense(Ax_minus_b, M_DIM, 1);
    float errorMetric = getL2NormDense(A_transpose_Ax_minus_b, N_DIM, 1) / denominator;
    printf("numerator: %lf\n", getL2NormDense(A_transpose_Ax_minus_b, N_DIM, 1));
    printf("denominator: %lf\n", denominator);
    printf("A_transpose_Ax_minus_b: ");
    for (int i = 0; i < N_DIM; i++) {
        printf("%lf ", A_transpose_Ax_minus_b[i]);
    }
    printf("\n");
    printf("Ax_minus_b: ");
    for (int i = 0; i < M_DIM; i++) {
        printf("%lf ", Ax_minus_b[i]);
    }
    printf("\n");
    printf("Ax: ");
    for (int i = 0; i < M_DIM; i++) {
        printf("%lf ", Ax[i]);
    }
    printf("\n");
    free(Ax);
    free(Ax_minus_b);
    free(A_transpose_Ax_minus_b);
    return errorMetric;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        return 1;
    }
    char* matrix_file = argv[1];
    float* sketched_a_matrix = (float*)calloc(D_DIM*N_DIM, sizeof(float));
    float* sketched_b_vec = (float*)calloc(D_DIM, sizeof(float));
    float* x_sketched = (float*)calloc(N_DIM, sizeof(float));
    float* x_sketched = (float*)calloc(N_DIM, sizeof(float));
    dsgemm_csc(sketched_a_matrix, sketching_matrix, a_matrix_indptr,a_matrix_indices, a_matrix_data, D_DIM, N_DIM, M_DIM);
    gemm(sketched_b_vec, sketching_matrix, b_vec, D_DIM, 1, M_DIM);
    x_sketched = householderQRLS(sketched_a_matrix, sketched_b_vec, D_DIM, N_DIM);
    // FILE *fptr;
    // char result_file[256];
    //    if (mkdir("../../dataset/results", 0777) == -1 && errno != EEXIST) {
    //     perror("Error creating directory");
    //     return 1;
    // }
    // sprintf(result_file, "../../dataset/results/%s.txt", matrix_file);
    // fptr = fopen("../../dataset/results/sk_32_fixed.txt", "w");
    // if (fptr == NULL)
    // {
    //     printf("Output file Error!");
    //     free(x_ls);
    //     free(sketched_a_matrix);
    //     free(sketched_b_vec);
    //     return 1;
    // } 
    // FILE *fptr;
    // char result_file[256];
    //    if (mkdir("../../dataset/results", 0777) == -1 && errno != EEXIST) {
    //     perror("Error creating directory");
    //     return 1;
    // }
    // sprintf(result_file, "../../dataset/results/%s.txt", matrix_file);
    // fptr = fopen("../../dataset/results/sk_32_fixed.txt", "w");
    // if (fptr == NULL)
    // {
    //     printf("Output file Error!");
    //     free(x_ls);
    //     free(sketched_a_matrix);
    //     free(sketched_b_vec);
    //     return 1;
    // } 
    printf("\n");
    printf("sketched result x:");
    // fprintf(fptr, "sketched result x:");
    // fprintf(fptr, "sketched result x:");
    for (int i = 0; i < N_DIM; i++) {
        printf("%lf ", x_sketched[i]);
        // fprintf(fptr, "%lf ", x_ls[i]);
    }
    printf("\n");
    // fprintf(fptr, "\n");
    printf("expected x:");
    // fprintf(fptr, "expected x:");
    // fprintf(fptr, "expected x:");
    for (int i = 0; i < N_DIM; i++) {
        printf("%lf ", x_vec[i]);
        // fprintf(fptr, "%lf ", x_vec[i]);
        // fprintf(fptr, "%lf ", x_vec[i]);
    }
    printf("\n");
    // fprintf(fptr, "\n");
    printf("Error Metric: %lf\n", getErrorMetric(x_sketched));
    // fclose(fptr);
    free(sketched_a_matrix);
    free(sketched_b_vec);
    free(x_sketched);
    return 0;
}