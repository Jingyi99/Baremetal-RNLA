#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include "gemm.c"
#include "riscv_vector.h"

// float house(int m, float* x, float* v);
// float* houseHolderHelper(float beta, float* v, int m);


// // textbook implementation
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

// // more intuitive implementation
// // float house(int m, float* x, float* v){
// //     float xNorm = 0;
// //     for (int i = 0; i < m; i++){
// //         xNorm += x[i] * x[i];
// //     }
// //     xNorm = sqrt(xNorm);
// //     float sign = (x[0] >= 0) ? 1 : -1;
// //     memcpy(v, x, m * sizeof(float));
// //     v[0] = x[0] + sign * xNorm;
// //     float vtv = 0;
// //     for (int i = 0; i < m; i++){
// //         vtv += v[i] * v[i];
// //     }
// //     float beta = 2 / vtv;
// //     return beta;
// // }

// void houseHolderQR(float* A, int m, int n){
//     for (int j = 0; j < n; j ++){
//         float *v = (float*)malloc((m-j)*sizeof(float));
//         float *x = (float*)malloc((m-j)*sizeof(float));
//         for (int i = j; i < m; i++){
//             x[i-j] = A[i*n+j];
//         }
//         float beta = house(m-j, x, v);
//         // update A
//         float* H = houseHolderHelper(beta, v, m-j);
//         // multiply by A submatrix which is m-j * n-j
//         float *A_sub = (float*)malloc((m-j)*(n-j)*sizeof(float));
//         float *A_sub_updated = (float*)malloc((m-j)*(n-j)*sizeof(float));
//         for (int i = j; i < m; i++){
//             for (int k = j; k < n; k++){
//                 A_sub[(i-j)*(n-j) + k-j] = A[i*n+k];
//             }
//         }
//         gemm(A_sub_updated, H, A_sub, m-j, n-j, m-j);
//         for (int i = j; i < m; i++){
//             for (int k = j; k < n; k++){
//                 A[i*n+k] = A_sub_updated[(i-j)*(n-j) + k-j];
//             }
//         }
//         free(A_sub);
//         free(A_sub_updated);
//         free(H);
//         free(v);
//         free(x);
//     }
// }

// // generate I - beta * v * vT
// float* houseHolderHelper(float beta, float* v, int m){
//     float* result = (float*)malloc(m*m*sizeof(float));
//     float* idenityMatrix = (float*)malloc(m*m*sizeof(float));
//     for (int i = 0; i < m; i++){
//         for (int j = 0; j < m; j++){
//             if (i == j){
//                 idenityMatrix[i*m+j] = 1;
//             } else {
//                 idenityMatrix[i*m+j] = 0;
//             }
//         }
//     }
//     float* vvT = (float*)malloc(m*m*sizeof(float));
//     gemm(vvT, v, v, m, m, 1);
//     for (int i = 0; i < m; i++){
//         for (int j = 0; j < m; j++){
//             result[i*m+j] = idenityMatrix[i*m+j] - beta * vvT[i*m+j];
//         }
//     }
//     return result;
// }

void backSubstitution(float* R, float* y, float* x, int m, int n){
    float sum;
    size_t vl;
    int32_t i;
    int32_t j;
    int32_t k;

    vfloat32m1_t R_p;
    vfloat32m1_t x_p;
    vfloat32m1_t prod;
    vfloat32m1_t sum_v;

    // First iteration
    x[m-1] = y[m-1] / R[(m-1)*n+(n-1)];

    j = 1;
    for (i = m-2; i >= 0; i--) {
        k   = j; // also m-i-1

        // Zero sum vector (splat)
        sum_v = __riscv_vfmv_v_f_f32m1(0.0, 1);
        while (k > 0) {
            vl    = __riscv_vsetvl_e32m1(k);
            printf("i: %d \tvl: %d\t", i, vl);
            printf("\tR index: %d\n", (i*n) + (n-k));
            // Load vectors
            x_p   = __riscv_vle32_v_f32m1(x + i + 1 + (j-k), vl);
            R_p   = __riscv_vle32_v_f32m1(R + (i*n) + (n-k), vl);
            // Dot Product
            prod  = __riscv_vfmul_vv_f32m1(R_p, x_p, vl);
            sum_v = __riscv_vfredosum_vs_f32m1_f32m1(prod, sum_v, vl);
            k    -= vl;
        }

        sum  = __riscv_vfmv_f_s_f32m1_f32(sum_v);
        x[i] = (y[i] - sum) / R[i*n+i];
        j++;
    }


    // for (int i = n-1; i >= 0; i--){
    //     float sum = 0;
    //     for (int j = i+1; j < n; j++){
    //         sum += R[i*n+j] * x[j];
    //     }
    //     x[i] = (y[i] - sum) / R[i*n+i];
    // }
    // return x;
}

// void houseHolderQRb(float* A, float* b, int m, int n) {
//     for (int j = 0; j < n; j ++){
//         float *v = (float*)malloc((m-j)*sizeof(float));
//         float *x = (float*)malloc((m-j)*sizeof(float));
//         for (int i = j; i < m; i++){
//             x[i-j] = A[i*n+j];
//         }
//         float beta = house(m-j, x, v);
//         // update A
//         float* H = houseHolderHelper(beta, v, m-j);
//         float* b_sub = (float*) malloc((m-j)*sizeof(float));
//         float* b_sub_updated = (float*) malloc((m-j)*sizeof(float));
//         memcpy(b_sub, &b[j], sizeof(float)*(m-j));
//         gemm(b_sub_updated, H, b_sub, m-j, 1, m-j);
//         for (int i = j; i < m; i++){
//             b[i] = b_sub_updated[i-j];
//         }
//         // memcpy(&b[j], b_sub, sizeof(float)*(m-j));
//         // multiply by A submatrix which is m-j * n-j
//         float *A_sub = (float*)malloc((m-j)*(n-j)*sizeof(float));
//         float *A_sub_updated = (float*)malloc((m-j)*(n-j)*sizeof(float));
//         for (int i = j; i < m; i++){
//             for (int k = j; k < n; k++){
//                 A_sub[(i-j)*(n-j) + k-j] = A[i*n+k];
//             }
//         }
//         gemm(A_sub_updated, H, A_sub, m-j, n-j, m-j);
//         for (int i = j; i < m; i++){
//             for (int k = j; k < n; k++){
//                 A[i*n+k] = A_sub_updated[(i-j)*(n-j) + k-j];
//             }
//         }
//         free(A_sub);
//         free(A_sub_updated);
//         free(H);
//         free(v);
//         free(x);
//         free(b_sub);
//         free(b_sub_updated);
//     }
// }

// float* householderQRLS(float* A, float* b, int m, int n){
//     houseHolderQRb(A, b, m, n);
//     // now b is updated to QTb and A is updated to R
//     // solve Rx = Q^Tb

//     return backSubstitution(A, b, m, n);
// }



// void test_house() {
//     // set test cases here
//     int m = 3;
//     float x[] = {2, 2, 1};
//     float v[] = {0, 0, 0};
//     //
//     float beta = house(m, x, v);
//     for (int i = 0; i < m; i++) {
//         printf("x: %f, v: %f\n", x[i], v[i]);
//     }
//     printf("beta: %f\n", beta);
// }

// void test_houseHolderHelper() {
//     // set test case here
//     int m = 3;
//     float x[] = {2, 2, 1};
//     // float v[] = {5, 2, 1};
//     // float beta = 2.0f / 30.0f;
//     float v[] = {1, -2, -1};
//     float beta = 1.0f / 3.0f;
//     //
//     float* result = houseHolderHelper(beta, v, m);

//     printf("H:\n");
//     for (int i = 0; i < m; i++){
//         for (int j = 0; j < m; j++){
//             printf("%f ", result[i*m+j]);
//         }
//         printf("\n");
//     }
//     float *Hx = malloc(sizeof(float) * m);
//     gemm(Hx, result, x, m, 1, m);
//     printf("Hx: \n");
//     for (int i = 0; i < m; i++){
//         printf("%f ", Hx[i]);
//     }
//     printf("\n");
//     free(Hx);
// }

// void test_houseHolderQR() {
//     // set test case here
//     // int m = 4;
//     // int n = 3;
//     // float A[] = {1.0f, -1.0f, 4.0f, 1.0f, 4.0f, -2.0f, 1.0f, 4.0f, 2.0f, 1.0f, -1.0f, 0.0f};
//     // int m = 3;
//     // int n = 3;
//     // float A[] = {2.0f, -2.0f, 18.0f, 2.0f, 1.0f, 0.0f, 1.0f, 2.0f, 0.0f};
//     int m = 3;
//     int n = 2;
//     float A[] = {1.0, -4.0, 2.0, 3.0, 2.0, 2.0};
//     printf("A:\n");
//     for (int i = 0; i < m; i++){
//         for (int j = 0; j < n; j++){
//             printf("%f ", A[i*n+j]);
//         }
//         printf("\n");
//     }
//     houseHolderQR(A, m, n);
//     printf("R:\n");
//     for (int i = 0; i < m; i++){
//         for (int j = 0; j < n; j++){
//             printf("%f ", A[i*n+j]);
//         }
//         printf("\n");
//     }
// }

void test_backSubstitution() {
    // set test case here
    int m = 3;
    int n = 3;
    float R[] = {1.0f, -2.0f, 1.0f,
                 0.0f, 1.0f, 6.0f,
                 0.0f, 0.0f, 1.0f};
    float y[] = {4.0f, -1.0f, 2.0f};
    float x[3] = {0.0};

    printf("Calling backsubstition\n");
    backSubstitution(R, y, x, m, n);

    printf("x: \n");
    for (int i = 0; i < n; i++) {
        printf("%.3f ", x[i]);
    }
    printf("\n");

    // Test case 2
    m = 10;
    n = 10;
    float R1[10][10] = {{0.328406, 0.592807, 0.963605, 0.047914, 0.583902, 0.663277, 0.821321, 0.467992, 0.441245, 0.869632},
                        {0.000000, 0.287301, 0.810943, 0.594004, 0.299508, 0.506638, 0.629284, 0.176319, 0.036821, 0.382882},
                        {0.000000, 0.000000, 0.763557, 0.872771, 0.598384, 0.811528, 0.195214, 0.498635, 0.846730, 0.788322},
                        {0.000000, 0.000000, 0.000000, 0.097213, 0.924739, 0.795865, 0.976783, 0.776233, 0.022980, 0.135772},
                        {0.000000, 0.000000, 0.000000, 0.000000, 0.793991, 0.748543, 0.093151, 0.347933, 0.043096, 0.423781},
                        {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.656377, 0.284855, 0.594453, 0.233624, 0.298250},
                        {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.105906, 0.326819, 0.336461, 0.597929},
                        {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.722163, 0.119044, 0.746278},
                        {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.246312, 0.922411},
                        {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.592532}};
    float y1[10] = {9, -10,   2,   1, -10,  -3,   4, -15, -15,  -6};
    float x1[10] = {0.0};    
    float x1_gold[10] = {-1412.0336846940822, -2328.4704431714517, 1889.3309535190674, -1621.1434792606606, 38.486833695540504, -67.49929122866932, 188.0552272912475, -6.519084264460278, -22.977492873375027, -10.126042145412656};         

    printf("\nTest Case 2: %dx%d\n", m, n);
    printf("Calling backsubstition\n");
    backSubstitution(R1[0], y1, x1, m, n);

    printf("x: \n");
    bool pass = true;
    for (int i = 0; i < n; i++) {
        printf("%.3f ", x1[i]);
        pass |= (x1[i]-x1_gold[i]) < 1e-4;
    }
    printf("\n");
    printf(pass ? "PASS\n" : "FAIL\n");


}
// void test_houseHolderQRLS() {
//     int m = 3;
//     int n = 2;
//     float A[] = {1.0, -4.0, 2.0, 3.0, 2.0, 2.0};
//     float b[] = {1.0, 2.0, 1.0};

//     float *x = (float*) malloc(sizeof(float) * n);
//     x = householderQRLS(A, b, m, n);
//     printf("x: \n");
//     for (int i = 0; i < n; i++) {
//         printf("%f ", x[i]);
//     }
//     printf("\n");
// }

int main() {
    // test_house();
    // test_houseHolderHelper();
    // test_houseHolderQR();
    test_backSubstitution();
    // test_houseHolderQRLS();
}
