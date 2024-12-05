#include <stdio.h>
#include <math.h>
#include <string.h>
#include "gemm.c"

float house(int m, float* x, float* v);
float* houseHolderHelper(float beta, float* v, int m);

// float house(int m, float* x, float* v) {
//     float sigma = 0;
//     float beta = 0;
//     for (int i = 1; i < m; i++) {
//         sigma += x[i] * x[i];
//     }
//     memcpy(v, x, m * sizeof(float));
//     v[0] = 1;
//     if (sigma == 0 && x[0] >= 0) {
//         beta = 0.0;
//     } else if (sigma == 0 && x[0] < 0) {
//         beta = -2.0;
//     } else {
//         float mu = sqrt(x[0]*x[0] + sigma);
//         if (x[0] <= 0) {
//             v[0] = x[0] - mu;
//         } else {
//             v[0] = -sigma / (x[0] + mu);
//         }
//         beta = 2 * v[0] * v[0] / (sigma + v[0]*v[0]);
//         float v0 = v[0];
//         if (v0 != 0) {
//             for (int i = 0; i < m; i++) {
//                 v[i] = v[i]/v0;
//             }
//         }
//         else {
//             for (int i = 0; i < m; i++) {
//                 v[i] = 0;
//             }
//         }
//     }
//     return beta;
// }

float house(int m, float* x, float* v){
    float xNorm = 0;
    for (int i = 0; i < m; i++){
        xNorm += x[i] * x[i];
    }
    xNorm = sqrt(xNorm);
    float e1[m];
    memset(e1, 0, m * sizeof(float));
    e1[0] = 1;
    float sign = (x[0] >= 0) ? 1 : -1;
    for (int i = 0; i < m; i++){
        v[i] = x[i] + sign * xNorm * e1[i];
    }
    float vtv = 0;
    for (int i = 0; i < m; i++){
        vtv += v[i] * v[i];
    }
    float beta = 2 / vtv;
    return beta;
}

// // A: m x n
void houseHolderQR(float* A, int m, int n){
    for (int j = 0; j < n; j ++){
        float *v = (float*)malloc((m-j)*sizeof(float));
        float *x = (float*)malloc((m-j)*sizeof(float));
        for (int i = j; i < m; i++){
            x[i-j] = A[i*n+j];
        } 
        float beta = house(m-j, x, v);
        // update A
        float* H = houseHolderHelper(beta, v, m-j);
        // need to multiply by A submatrix which is m-j x n-j
        float *A_sub = (float*)malloc((m-j)*(n-j)*sizeof(float));
        float *A_sub_updated = (float*)malloc((m-j)*(n-j)*sizeof(float));
        for (int i = j; i < m; i++){
            for (int k = j; k < n; k++){
                A_sub[(i-j)*(n-j) + k-j] = A[i*n+k];
            }
        }
        gemm(A_sub_updated, H, A_sub, m-j, n-j, m-j);
            printf("A_sub_updated:\n");
        for (int ii = 0; ii < m; ii++){
            for (int jj = 0; jj < n; jj++){
                printf("%f ", A_sub_updated[ii*n+jj]);
            }
            printf("\n");
        }
        for (int i = j; i < m; i++){
            for (int k = j; k < n; k++){
                A[i*n+k] = A_sub_updated[(i-j)*(n-j) + k-j];
            }
        }
        free(A_sub);
        free(H);
        free(A_sub_updated);
    }
}

float* houseHolderHelper(float beta, float* v, int m){
    float* result = (float*)malloc(m*m*sizeof(float));
    float* idenityMatrix = (float*)malloc(m*m*sizeof(float));
    for (int i = 0; i < m; i++){
        for (int j = 0; j < m; j++){
            if (i == j){
                idenityMatrix[i*m+j] = 1;
            } else {
                idenityMatrix[i*m+j] = 0;
            }
        }
    }
    float* vvT = (float*)malloc(m*m*sizeof(float));
    gemm(vvT, v, v, m, m, 1);
    // print out vvT
    // printf("vvT:\n");
    // for (int i = 0; i < m; i++){
    //     for (int j = 0; j < m; j++){
    //         printf("%f ", vvT[i*m+j]);
    //     }
    //     printf("\n");
    // }
    // printf("beta: %f\n", beta);
    for (int i = 0; i < m; i++){
        for (int j = 0; j < m; j++){
            // printf("i: %d, j:%d, vvt:%f\n", i, j, beta * vvT[i*m+j]);
            result[i*m+j] = idenityMatrix[i*m+j] - beta * vvT[i*m+j];
        }
    }
    return result;
}

void test_house() {
    float x[3] = {1, 2, 2};
    float v[3] = {0, 0, 0};
    float beta = house(3, x, v);
    for (int i = 0; i < 3; i++) {
        printf("x: %f, v: %f\n", x[i], v[i]);
    }
    printf("beta: %f\n", beta);
}

void test_houseHolderHelper() {
    float v[3] = {4, 2, 2};
    float beta = 0.0833;
    float* result = houseHolderHelper(beta, v, 3);

    printf("H:\n");
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            printf("%f ", result[i*3+j]);
        }
        printf("\n");
    }
}

void test_houseHolder() {
    // float A[12] = {1.0f, -1.0f, 4.0f, 1.0f, 4.0f, -2.0f, 1.0f, 4.0f, 2.0f, 1.0f, -1.0f, 0.0f};
    int m = 3;
    int n = 2;
    float A[6] = {1.0, -4.0, 2.0, 3.0, 2.0, 2.0};

    houseHolderQR(A, m, n);
    printf("R:\n");
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            printf("%f ", A[i*n+j]);
        }
        printf("\n");
    }
}

int main() {
    test_house();
    test_houseHolderHelper();
    test_houseHolder();
}
