#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
// #include "gemm.c"
#include "gemm_rvv.h"

double house(int m, double* x, double* v);
double* houseHolderHelper(double beta, double* v, int m);
extern void gemm(double *c_matrix, const double *a_matrix, const double *b_matrix, const unsigned int m_dim, const unsigned int n_dim,  const unsigned int k_dim);

// textbook implementation
double house(int m, double* x, double* v) {
    double sigma = 0.0;
    double beta = 0.0;
    for (int i = 1; i < m; i++) {
        sigma += x[i] * x[i];
    }
    memcpy(v, x, m * sizeof(double));
    v[0] = 1;
    if (sigma == 0 && x[0] >= 0) {
        beta = 0.0;
    } else if (sigma == 0 && x[0] < 0) {
        beta = -2.0;
    } else {
        double mu = sqrtf(x[0]*x[0] + sigma);
        if (x[0] <= 0) {
            v[0] = x[0] - mu;
        } else {
            v[0] = -sigma / (x[0] + mu);
        }
        beta = 2 * v[0] * v[0] / (sigma + v[0]*v[0]);
        double v0 = v[0];
        if (v0 != 0) {
            for (int i = 0; i < m; i++) {
                v[i] = v[i]/v0;
            }
        }
        else {
            for (int i = 0; i < m; i++) {
                v[i] = 0;
            }
        }
    }
    return beta;
}

// more intuitive implementation
// double house(int m, double* x, double* v){
//     double xNorm = 0;
//     for (int i = 0; i < m; i++){
//         xNorm += x[i] * x[i];
//     }
//     xNorm = sqrt(xNorm);
//     double sign = (x[0] >= 0) ? 1 : -1;
//     memcpy(v, x, m * sizeof(double));
//     v[0] = x[0] + sign * xNorm;
//     double vtv = 0;
//     for (int i = 0; i < m; i++){
//         vtv += v[i] * v[i];
//     }
//     double beta = 2 / vtv;
//     return beta;
// }

void houseHolderQR(double* A, int m, int n){
    for (int j = 0; j < n; j ++){
        double *v = (double*)malloc((m-j)*sizeof(double));
        double *x = (double*)malloc((m-j)*sizeof(double));
        for (int i = j; i < m; i++){
            v[i-j] = A[i*n+j];
        }
        for (int i = j; i < m; i++){
            x[i-j] = A[i*n+j];
        }
        double beta = house(m-j, x, v);
        double* H = houseHolderHelper(beta, v, m-j);
        // multiply by A submatrix which is m-j * n-j
        double *A_sub = (double*)malloc((m-j)*(n-j)*sizeof(double));
        double *A_sub_updated = (double*)malloc((m-j)*(n-j)*sizeof(double));
        for (int i = j; i < m; i++){
            for (int k = j; k < n; k++){
                A_sub[(i-j)*(n-j) + k-j] = A[i*n+k];
            }
        }
        gemm(A_sub_updated, H, A_sub, m-j, n-j, m-j);
        for (int i = j; i < m; i++){
            for (int k = j; k < n; k++){
                A[i*n+k] = A_sub_updated[(i-j)*(n-j) + k-j];
            }
        }
        free(A_sub);
        free(A_sub_updated);
        free(H);
        free(v);
        free(x);
    }
}

// generate I - beta * v * vT
double* houseHolderHelper(double beta, double* v, int m){
    double* result = (double*)malloc(m*m*sizeof(double));
    double* idenityMatrix = (double*)malloc(m*m*sizeof(double));
    for (int i = 0; i < m; i++){
        for (int j = 0; j < m; j++){
            if (i == j){
                idenityMatrix[i*m+j] = 1;
            } else {
                idenityMatrix[i*m+j] = 0;
            }
        }
    }
    double* vvT = (double*)malloc(m*m*sizeof(double));
    gemm(vvT, v, v, m, m, 1);
    for (int i = 0; i < m; i++){
        for (int j = 0; j < m; j++){
            result[i*m+j] = idenityMatrix[i*m+j] - beta * vvT[i*m+j];
        }
    }
    return result;
}

double* backSubstitution(double* R, double* y, int m, int n){
    if (m < n) {
        fprintf(stderr, "Invalid matrix dimensions: m must be >= n\n");
        exit(EXIT_FAILURE);
    }
    double* x = (double*)malloc(n*sizeof(double));
    for (int i = n-1; i >= 0; i--){
        double sum = 0;
        for (int j = i+1; j < n; j++){
            sum += R[i*n+j] * x[j];
        }
        if (R[i * n + i] == 0.0) {
            fprintf(stderr, "Division by zero at row %d\n", i);
            free(x); // Clean up allocated memory
            exit(EXIT_FAILURE);
        }
        x[i] = (y[i] - sum) / R[i*n+i];
    }
    return x;
}

void houseHolderQRb(double* A, double* b, int m, int n) {
    for (int j = 0; j < n; j ++){
        double *v = (double*)malloc((m-j)*sizeof(double));
        double *x = (double*)malloc((m-j)*sizeof(double));
        for (int i = j; i < m; i++){
            v[i-j] = A[i*n+j];
        }
        for (int i = j; i < m; i++){
            x[i-j] = A[i*n+j];
        }
        double beta = house(m-j, x, v);
        double* H = houseHolderHelper(beta, v, m-j);
        double* b_sub = (double*) malloc(( m-j) * sizeof(double));
        double* b_sub_updated = (double*) malloc((m-j)*sizeof(double));
        memcpy(b_sub, &b[j], sizeof(double)*(m-j));
        gemm(b_sub_updated, H, b_sub, m-j, 1, m-j);
        for (int i = j; i < m; i++){
            b[i] = b_sub_updated[i-j];
        }
        // memcpy(&b[j], b_sub, sizeof(double)*(m-j));
        // multiply by A submatrix which is m-j * n-j
        double *A_sub = (double*)malloc((m-j)*(n-j)*sizeof(double));
        double *A_sub_updated = (double*)malloc((m-j)*(n-j)*sizeof(double));
        for (int i = j; i < m; i++){
            for (int k = j; k < n; k++){
            A_sub[(i-j)*(n-j) + k-j] = A[i*n+k];
            }
        }
        gemm(A_sub_updated, H, A_sub, m-j, n-j, m-j);
        for (int i = j; i < m; i++){
            for (int k = j; k < n; k++){
            A[i*n+k] = A_sub_updated[(i-j)*(n-j) + k-j];
            }
        }
        free(A_sub);
        free(A_sub_updated);
        free(H);
        free(v);
        free(x);
        free(b_sub);
        free(b_sub_updated);
        }
    }

    double* householderQRLS(double* A, double* b, int m, int n){
        houseHolderQRb(A, b, m, n);
        printf("A: \n");
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                printf("%f ", A[i*n+j]);
            }
            printf("\n");
        }
        // print b
        printf("b: \n");
        for (int i = 0; i < m; i++){
            printf("%f ", b[i]);
        }
        // now b is updated to QTb and A is updated to R
        // solve Rx = Q^Tb
        return backSubstitution(A, b, m, n);
    }

    // TODO: Move tests to another file
    void test_house() {
        // set test cases here
        int m = 3;
        double x[] = {2, 2, 1};
        double v[] = {0, 0, 0};
        //
        double beta = house(m, x, v);
        for (int i = 0; i < m; i++) {
        printf("x: %f, v: %f\n", x[i], v[i]);
        }
        printf("beta: %f\n", beta);
    }

    void test_houseHolderHelper() {
        // set test case here
        int m = 3;
        double x[] = {2, 2, 1};
        // double v[] = {5, 2, 1};
        // double beta = 2.0 / 30.0;
        double v[] = {1, -2, -1};
        double beta = 1.0 / 3.0;
        //
        double* result = houseHolderHelper(beta, v, m);

        printf("H:\n");
        for (int i = 0; i < m; i++){
        for (int j = 0; j < m; j++){
            printf("%f ", result[i*m+j]);
        }
        printf("\n");
        }
        double *Hx = malloc(sizeof(double) * m);
        gemm(Hx, result, x, m, 1, m);
        printf("Hx: \n");
        for (int i = 0; i < m; i++){
        printf("%f ", Hx[i]);
        }
        printf("\n");
        free(Hx);
    }

    void test_houseHolderQR() {
        // set test case here
        // int m = 4;
        // int n = 3;
        // double A[] = {1.0, -1.0, 4.0, 1.0, 4.0, -2.0, 1.0, 4.0, 2.0, 1.0, -1.0, 0.0};
        // int m = 3;
        // int n = 3;
        // double A[] = {2.0, -2.0, 18.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0};
        int m = 4;
        int n = 3;
        double A[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 10.0, 11.0, 13.0};
        printf("A:\n");
        for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            printf("%f ", A[i*n+j]);
        }
        printf("\n");
        }
        houseHolderQR(A, m, n);
        printf("R:\n");
        for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            printf("%f ", A[i*n+j]);
        }
        printf("\n");
        }
    }

    void test_backSubstitution() {
        int m = 4;
        int n = 3;
        double A[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 10.0, 11.0, 13.0};
        double b[] = {14.0, 32.0, 50.0, 68.0};
        double R[] = {12.884099 , 14.591630 , 16.299161 , 0.0, 1.041315 , 2.082630 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double y[] = {90.964842, 8.330522 , 0.000000 , 0.000000 };
        householderQRLS(A, b, m, n);
        printf("A back: \n");
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                printf("%f ", A[i*n+j]);
            }
            printf("\n");
        }
        printf("b back: \n");
        for (int i = 0; i < m; i++){
            printf("%f ", b[i]);
        }
        double *x = backSubstitution(A, b, m, n);
        printf("x: \n");
        for (int i = 0; i < n; i++) {
        printf("%f ", x[i]);
        }
        printf("\n");
    }

    void test_houseHolderQRLS() {
        int m = 4;
        int n = 3;
        double A[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 10.0, 11.0, 13.0};
        double b[] = {14.0, 32.0, 50.0, 68.0};

        double *x = (double*) malloc(sizeof(double) * n);
        memset(x, 0, sizeof(double) * n);
        x = householderQRLS(A, b, m, n);
        printf("x: \n");
        for (int i = 0; i < n; i++) {
        printf("%lf ", x[i]);
        }
        printf("\n");
    }

     int main() {
         // test_house();
         // test_houseHolderHelper();
         // test_houseHolderQR();
         // test_backSubstitution();
         test_houseHolderQRLS();
     }
