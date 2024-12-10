#include <stdio.h>
#include <math.h>
#include <string.h>
#include "gemm.c"

double house(int m, double* x, double* v);
double* houseHolderHelper(double beta, double* v, int m);
void test_backwarderror(double* A, double* Q, double* R, int m, int n);

double house(int m, double* x, double* v) {
    double sigma = 0.0;
    double beta = 0.0;
    for (int i = 1; i < m; i++) {
        sigma += x[i] * x[i];
    }
    v[0] = 1; 
    if (sigma == 0 && x[0] >= 0) {
        beta = 0.0;
    } else if (sigma == 0 && x[0] < 0) {
        beta = -2.0;
    } else {
        double mu = sqrt(x[0]*x[0] + sigma);
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
    }
    return beta;
}

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
//     double v0 = v[0];
//     for (int i = 0; i < m; i++){
//         v[i] = v[i]/v0;
//         vtv += v[i] * v[i];
//     }
//     double beta = 2 / vtv;
//     return beta;
// }

void houseHolderQR(double* A, int m, int n){
    for (int j = 0; j < n; j ++){
        double *v = (double*)calloc((m-j), sizeof(double));
        double *x = (double*)calloc((m-j), sizeof(double));
        for (int i = j; i < m; i++){
            v[i-j] = A[i*n+j];
        }
        for (int i = j; i < m; i++){
            x[i-j] = A[i*n+j];
        }
        double beta = house(m-j, x, v);
        double* H = houseHolderHelper(beta, v, m-j);
        // multiply by A submatrix which is m-j * n-j
        double *A_sub = (double*)calloc((m-j)*(n-j), sizeof(double));
        double *A_sub_updated = (double*)calloc((m-j)*(n-j), sizeof(double));
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
        if (j < m) {
            for (int k = j+1; k < m; k++) {
                A[k*n+j] = v[k-j];
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
    double* result = (double*)calloc(m*m, sizeof(double));
    double* identityMatrix = (double*)calloc(m*m, sizeof(double));
    for (int i = 0; i < m; i++) {
        identityMatrix[i*m+i] = 1.0;
    }
    double* vvT = (double*)calloc(m*m, sizeof(double));
    gemm(vvT, v, v, m, m, 1);
    for (int i = 0; i < m; i++){
        for (int j = 0; j < m; j++){
            result[i*m+j] = identityMatrix[i*m+j] - beta * vvT[i*m+j];
        }
    }
    free(identityMatrix);
    free(vvT);
    return result;
}

double* backSubstitution(double* R, double* y, int m, int n){
    if (m < n) {
        fprintf(stderr, "Invalid matrix dimensions: m must be >= n\n");
        exit(EXIT_FAILURE);
    }
    double* x = (double*)calloc(n, sizeof(double));
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
        double *v = (double*)calloc((m-j), sizeof(double));
        double *x = (double*)calloc((m-j), sizeof(double));
        for (int i = j; i < m; i++){
            v[i-j] = A[i*n+j];
        }
        for (int i = j; i < m; i++){
            x[i-j] = A[i*n+j];
        }
        double beta = house(m-j, x, v);
        double* H = houseHolderHelper(beta, v, m-j);
        double* b_sub = (double*) calloc(m-j, sizeof(double));
        double* b_sub_updated = (double*) calloc((m-j), sizeof(double));
        for (int i = 0; i < m-j; i++){
            b_sub[i] = b[i+j];
        }
        // vt*b
        double vtb = 0.0;
        for (int i = j; i < m; i++) {
            vtb += v[i-j] * b[i];
        }
        for (int i = j; i < m; i++) {
            b[i] = b[i] - beta*vtb*v[i-j];
        }
        // gemm(b_sub_updated, H, b_sub, m-j, 1, m-j);
        // for (int i = j; i < m; i++){
        //     b[i] = b_sub_updated[i-j];
        // } 
        // multiply by A submatrix which is m-j * n-j
        double *A_sub = (double*)calloc((m-j)*(n-j), sizeof(double));
        double *A_sub_updated = (double*)calloc((m-j)*(n-j), sizeof(double));
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
        // debugging
        // printf("A one col should be zeroed out:\n");
        // for (int ii = 0; ii < m; ii++) {
        //     for (int jj = 0; jj < n; jj++) {
        //         printf("%lf ", A[ii*n+jj]);
        //     }
        //     printf("\n");
        // }
        // printf("\n");

        free(A_sub);
        free(A_sub_updated);
        free(H);
        free(v);
        free(x);
        free(b_sub);
        free(b_sub_updated);
        }
    }

    // double* householderQRLS(double* A, double* b, int m, int n){
    //     houseHolderQR(A, m, n);
    //     for (int j = 0; j < n; j++) {
    //         double *v = malloc(sizeof(double) * (m-j));
    //         v[0] = 1.0;
    //         for (int k = j+1; k < m; k++) {
    //             v[k-j] = A[k*n+j];
    //         }
    //         double vtv = 0.0;
    //         for (int i = 0; i < m-j; i++) {
    //             vtv += v[i] * v[i];
    //         }
    //         double beta = 2.0/vtv;
    //         double vtb = 0.0;
    //         for (int i = j; i < m; i++) {
    //             vtb += v[i-j] * b[i];
    //         }
    //         for (int i = j; i < m; i++) {
    //             b[i] = b[i] - beta*vtb*v[i-j];
    //         }
    //     }
    //     return backSubstitution(A, b, m, n);
    // }


    double* householderQRLS(double* A, double* b, int m, int n){
        houseHolderQRb(A, b, m, n);
        // now b is updated to QTb and A is updated to R
        // solve Rx = Q^Tb
        return backSubstitution(A, b, m, n);
    }
