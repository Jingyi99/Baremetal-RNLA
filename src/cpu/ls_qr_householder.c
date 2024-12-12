#include <stdio.h>
#include <math.h>
#include <string.h>
#include "gemm.c"

float house(int m, float* x, float* v);
float* houseHolderHelper(float beta, float* v, int m);
void test_backwarderror(float* A, float* Q, float* R, int m, int n);

float house(int m, float* x, float* v) {
    float sigma = 0.0;
    float beta = 0.0;
    for (int i = 1; i < m; i++) {
        sigma += x[i] * x[i];
    }
    v[0] = 1; 
    if (sigma == 0 && x[0] >= 0) {
        beta = 0.0;
    } else if (sigma == 0 && x[0] < 0) {
        beta = -2.0;
    } else {
        float mu = sqrt(x[0]*x[0] + sigma);
        if (x[0] <= 0) {
            v[0] = x[0] - mu;
        } else {
            v[0] = -sigma / (x[0] + mu);
        }
        beta = 2 * v[0] * v[0] / (sigma + v[0]*v[0]);
        float v0 = v[0];
        if (v0 != 0) {
            for (int i = 0; i < m; i++) {
                v[i] = v[i]/v0;
            }
        }
    }
    return beta;
}

// float house(int m, float* x, float* v){
//     float xNorm = 0;
//     for (int i = 0; i < m; i++){
//         xNorm += x[i] * x[i];
//     }
//     xNorm = sqrt(xNorm);
//     float sign = (x[0] >= 0) ? 1 : -1;
//     memcpy(v, x, m * sizeof(float));
//     v[0] = x[0] + sign * xNorm;
//     float vtv = 0;
//     float v0 = v[0];
//     for (int i = 0; i < m; i++){
//         v[i] = v[i]/v0;
//         vtv += v[i] * v[i];
//     }
//     float beta = 2 / vtv;
//     return beta;
// }

void houseHolderQR(float* A, int m, int n){
    for (int j = 0; j < n; j ++){
        float *v = (float*)calloc((m-j), sizeof(float));
        float *x = (float*)calloc((m-j), sizeof(float));
        for (int i = j; i < m; i++){
            v[i-j] = A[i*n+j];
        }
        for (int i = j; i < m; i++){
            x[i-j] = A[i*n+j];
        }
        float beta = house(m-j, x, v);
        float* H = houseHolderHelper(beta, v, m-j);
        // multiply by A submatrix which is m-j * n-j
        float *A_sub = (float*)calloc((m-j)*(n-j), sizeof(float));
        float *A_sub_updated = (float*)calloc((m-j)*(n-j), sizeof(float));
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
float* houseHolderHelper(float beta, float* v, int m){
    float* result = (float*)calloc(m*m, sizeof(float));
    float* identityMatrix = (float*)calloc(m*m, sizeof(float));
    for (int i = 0; i < m; i++) {
        identityMatrix[i*m+i] = 1.0;
    }
    float* vvT = (float*)calloc(m*m, sizeof(float));
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

float* backSubstitution(float* R, float* y, int m, int n){
    if (m < n) {
        fprintf(stderr, "Invalid matrix dimensions: m must be >= n\n");
        exit(EXIT_FAILURE);
    }
    float* x = (float*)calloc(n, sizeof(float));
    for (int i = n-1; i >= 0; i--){
        float sum = 0;
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

void houseHolderQRb(float* A, float* b, int m, int n) {
    for (int j = 0; j < n; j ++){
        float *v = (float*)calloc((m-j), sizeof(float));
        float *x = (float*)calloc((m-j), sizeof(float));
        for (int i = j; i < m; i++){
            v[i-j] = A[i*n+j];
        }
        for (int i = j; i < m; i++){
            x[i-j] = A[i*n+j];
        }
        float beta = house(m-j, x, v);
        float* H = houseHolderHelper(beta, v, m-j);
        float* b_sub = (float*) calloc(m-j, sizeof(float));
        float* b_sub_updated = (float*) calloc((m-j), sizeof(float));
        for (int i = 0; i < m-j; i++){
            b_sub[i] = b[i+j];
        }
        // vt*b
        float vtb = 0.0;
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
        float *A_sub = (float*)calloc((m-j)*(n-j), sizeof(float));
        float *A_sub_updated = (float*)calloc((m-j)*(n-j), sizeof(float));
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

    // float* householderQRLS(float* A, float* b, int m, int n){
    //     houseHolderQR(A, m, n);
    //     for (int j = 0; j < n; j++) {
    //         float *v = malloc(sizeof(float) * (m-j));
    //         v[0] = 1.0;
    //         for (int k = j+1; k < m; k++) {
    //             v[k-j] = A[k*n+j];
    //         }
    //         float vtv = 0.0;
    //         for (int i = 0; i < m-j; i++) {
    //             vtv += v[i] * v[i];
    //         }
    //         float beta = 2.0/vtv;
    //         float vtb = 0.0;
    //         for (int i = j; i < m; i++) {
    //             vtb += v[i-j] * b[i];
    //         }
    //         for (int i = j; i < m; i++) {
    //             b[i] = b[i] - beta*vtb*v[i-j];
    //         }
    //     }
    //     return backSubstitution(A, b, m, n);
    // }


    float* householderQRLS(float* A, float* b, int m, int n){
        houseHolderQRb(A, b, m, n);
        // now b is updated to QTb and A is updated to R
        // solve Rx = Q^Tb
        return backSubstitution(A, b, m, n);
    }
