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
    // memcpy(v, x, m * sizeof(double));
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
            // v[0] = x[0] + mu;
        }
        beta = 2 * v[0] * v[0] / (sigma + v[0]*v[0]);
        double v0 = v[0];
        if (v0 != 0) {
            for (int i = 0; i < m; i++) {
                v[i] = v[i]/v0;
            }
        }
        // else {
        //     for (int i = 0; i < m; i++) {
        //         v[i] = 0;
        //     }
        // }
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
    // print R 
    // printf("R:\n");
    // for (int i = 0; i < m; i++){
    //     for (int j = 0; j < n; j++){
    //         printf("%f ", R[i*n+j]);
    //     }
    //     printf("\n");
    // }
    // print y
    // printf("y:\n");
    // for (int i = 0; i < m; i++){
    //     printf("%f ", y[i]);
    // }
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
        // printf("row: %d\n x: %lf\n", i, x[i]);
    }
    return x;
}

void houseHolderQRb(double* A, double* b, int m, int n) {
    double* Acopy = (double*)malloc(m*n*sizeof(double));
    memcpy(Acopy, A, m*n*sizeof(double));
    for (int j = 0; j < n; j ++){
        double *v = (double*)malloc((m-j)*sizeof(double));
        double *x = (double*)malloc((m-j)*sizeof(double));
        for (int i = j; i < m; i++){
            v[i-j] = A[i*n+j];
        }
        for (int i = j; i < m; i++){
            x[i-j] = A[i*n+j];
        }
        printf("start of x: %d \n", j*n+j);
        // write v to file 
        FILE *f = fopen("v.txt", "w");
        if (f == NULL)
        {
            printf("Error opening file!\n");
            exit(1);
        }
        for (int i = j; i < m; i++){
            fprintf(f, "%f\n", A[i*n+j]);
        }
        fclose(f);
        double beta = house(m-j, x, v);
        printf("v: \n");
        for (int i = 0; i < m; i++) {
            printf("%lf ", v[i]);
        }
        printf("\n");
        double* H = houseHolderHelper(beta, v, m-j);
        double* b_sub = (double*) malloc(( m-j) * sizeof(double));
        double* b_sub_updated = (double*) malloc((m-j)*sizeof(double));
        // memcpy(b_sub, &b[j], sizeof(double)*(m-j));
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
        // printf("b sub:\n");
        // for (int i = 0; i < m-j; i++) {
        //     printf("%lf ", b_sub[i]);
        // }
        // printf("\n");
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
        printf("A one col should be zeroed out:\n");
        for (int ii = 0; ii < m; ii++) {
            for (int jj = 0; jj < n; jj++) {
                printf("%lf ", A[ii*n+jj]);
            }
            printf("\n");
        }
        printf("\n");

        
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
    //     printf("R:\n");
    //     for (int i = 0; i < m; i++) {
    //         for (int j = 0; j < n; j++) {
    //             printf("%lf ", A[i*n+j]);
    //         }
    //         printf("\n");
    //     }
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
    //     printf("\n");
    //     printf("updated b:\n");
    //     for (int i = 0; i < m; i++) {
    //         printf("%lf ", b[i]);
    //     }
    //     printf("\n");
    //     return backSubstitution(A, b, m, n);
    // }


    double* householderQRLS(double* A, double* b, int m, int n){
        houseHolderQRb(A, b, m, n);
        // now b is updated to QTb and A is updated to R
        // solve Rx = Q^Tb
        printf("R:\n");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                printf("%lf ", A[i*n+j]);
            }
            printf("\n");
        }
        printf("\n");
        printf("updated b:\n");
        for (int i = 0; i < m; i++) {
            printf("%lf ", b[i]);
        }
        printf("\n");
        return backSubstitution(A, b, m, n);
    }

    void test_backwarderror(double* A, double* Q, double* R, int m, int n){
        // print R 
        FILE *file = fopen("R_matrix.txt", "w");
        if (file == NULL) {
            fprintf(stderr, "Error opening file for writing\n");
            return;
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                fprintf(file, "%f ", R[i * n + j]);
            }
            fprintf(file, "\n");
        }
        fclose(file);
        // print Q
        file = fopen("Q_matrix.txt", "w");
        if (file == NULL) {
            fprintf(stderr, "Error opening file for writing\n");
            return;
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                fprintf(file, "%f ", Q[i * m + j]);
            }
            fprintf(file, "\n");
        }
        fclose(file);
        // print (A - qr) / A
        double* qr = (double*)malloc(m*n*sizeof(double));
        gemm(qr, Q, R, m, n, n);
        double error = 0;
        double residual = 0;
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                residual += (A[i*n+j] - qr[i*n+j]) * (A[i*n+j] - qr[i*n+j]);
            }
        }
        residual = sqrt(residual);
        printf("residual: %f\n", residual);
        double normA = 0;
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                normA += A[i*n+j] * A[i*n+j];
            }
        }
        normA = sqrt(normA);
        printf("normA: %f\n", normA);
        error = residual / normA;
        printf("backward error: %f\n", error);
    }
