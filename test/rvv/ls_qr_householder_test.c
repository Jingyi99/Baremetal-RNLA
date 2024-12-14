#include <lib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include "gemm_rvv.h"
#include "riscv_vector.h"
#include "householder.h"
// Include header with test vector
#include "small_ls.h"

// TESTING/DEBUG
void test_house() {
    // set test cases here
    int m = 3;
    float x[] = {2, 2, 1};
    float v[] = {0, 0, 0};
    //
    float beta = house(m, x, v);
    for (int i = 0; i < m; i++) {
        printf("x: %f, v: %f\n", x[i], v[i]);
    }
    printf("beta: %f\n", beta);
}

void test_houseHolderHelper() {
    // set test case here
    int m = 3;
    float x[] = {2, 2, 1};
    // float v[] = {5, 2, 1};
    // float beta = 2.0f / 30.0f;
    float v[] = {1, -2, -1};
    float beta = 1.0f / 3.0f;
    //
    float* result = houseHolderHelper(beta, v, m);

    printf("H:\n");
    for (int i = 0; i < m; i++){
        for (int j = 0; j < m; j++){
            printf("%f ", result[i*m+j]);
        }
        printf("\n");
    }
    float *Hx = calloc(m, sizeof(float));
    gemm_rvv(Hx, result, x, m, 1, m);
    printf("Hx: \n");
    for (int i = 0; i < m; i++){
        printf("%f ", Hx[i]);
    }
    printf("\n");
    free(Hx);
}

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

void test_houseHolderQRLS() {
    float *x = (float*) calloc(N_DIM, sizeof(float));
    x = householderQRLS(a_matrix, b_vec, M_DIM, N_DIM);

    printf("ref x: \n");
    for (int i = 0; i < N_DIM; i++) {
        printf("%f ", x_vec[i]);
    }
    printf("\n");
    printf("our x: \n");
    for (int i = 0; i < N_DIM; i++) {
        printf("%f ", x[i]);
    }
    printf("\n");
}
