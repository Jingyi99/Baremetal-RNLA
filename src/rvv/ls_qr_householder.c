#include <lib.h>
#include <math.h>
#include <stdio.h>
// #include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include "riscv_vector.h"
#include "gemm_rvv.h"

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

void* allocate_vector_clear(uint32_t bytes) {
    void* arr = malloc(bytes);
    size_t vl; //= __riscv_vsetvl_e8m8(1000); // Just to ensure VLMAX
    vuint8m8_t zero = __riscv_vmv_v_x_u8m8(0, vl);
    void* base = arr;
    for (int b=bytes; b > 0; b-=vl) {
        vl = __riscv_vsetvl_e8m8(b); // Just to ensure VLMAX
        __riscv_vse8_v_u8m8(base, zero, vl);
        base += vl;
    }
    return arr;
}


void vector_clear(void* arr, uint32_t bytes) {
    size_t vl; //= __riscv_vsetvl_e8m8(1000); // Just to ensure VLMAX
    vuint8m8_t zero = __riscv_vmv_v_x_u8m8(0, vl);
    void* base = arr;
    for (int b=bytes; b > 0; b-=vl) {
        vl = __riscv_vsetvl_e8m8(b); // Just to ensure VLMAX
        __riscv_vse8_v_u8m8(base, zero, vl);
        base += vl;
    }
}

static inline float dot_product(float* x, float* y, int m) {
    size_t vl;
    int32_t j = 0;
    vfloat32m1_t x_p;
    vfloat32m1_t y_p;
    vfloat32m1_t prod;
    vfloat32m1_t sum = __riscv_vfmv_v_f_f32m1(0.0, 1);

    for (int i = m; i > 0; i-=vl){
        vl = __riscv_vsetvl_e32m1(i);
        x_p = __riscv_vle32_v_f32m1(x + j, vl);
        y_p = __riscv_vle32_v_f32m1(y + j, vl);
        prod = __riscv_vfmul_vv_f32m1(x_p, y_p, vl);
        sum = __riscv_vfredosum_vs_f32m1_f32m1(prod, sum, vl);
        j += vl;
    }

    return __riscv_vfmv_f_s_f32m1_f32(sum);
}


// // more intuitive implementation
float house(int m, float* x, float* v){
    float xNorm = 0;
    size_t vl;
    int32_t i;
    int32_t j = 0;

    vfloat32m1_t x_p;
    vfloat32m1_t x_sum;
    vfloat32m1_t x_div;
    vfloat32m1_t x_prod;

    x_sum = __riscv_vfmv_v_f_f32m1(0.0, 1);
    for (i = m; i > 0; i-=vl){
        // xNorm += x[i] * x[i];
        vl = __riscv_vsetvl_e32m1(i);
        x_p = __riscv_vle32_v_f32m1(x + j, vl);
        x_prod = __riscv_vfmul_vv_f32m1(x_p, x_p, vl);
        x_sum = __riscv_vfredosum_vs_f32m1_f32m1(x_prod, x_sum, vl);
        j += vl;
    }
    // xNorm = __riscv_vfmv_f_s_f32m1_f32(x_sum);
    xNorm = dot_product(x, x, m);
    xNorm = sqrt(xNorm);
    float sign = (x[0] >= 0) ? 1 : -1;

    // memcpy(v, x, m * sizeof(float));
    x[0] = x[0] + sign * xNorm;
    float xtx = 0;
    float x0 = x[0];

    x_sum = __riscv_vfmv_v_f_f32m1(0.0, 1);
    j = 0;
    for (i = m; i > 0; i-=vl){
        // v[i] = v[i]/v0;
        // vtv += v[i] * v[i];
        vl = __riscv_vsetvl_e32m1(i);
        x_p = __riscv_vle32_v_f32m1(x + j, vl);
        x_div = __riscv_vfdiv_vf_f32m1(x_p, x0, vl);
        __riscv_vse32_v_f32m1(v + j, x_div, vl);
        x_prod = __riscv_vfmul_vv_f32m1(x_div, x_div, vl);
        x_sum = __riscv_vfredosum_vs_f32m1_f32m1(x_prod, x_sum, vl);
        j += vl;
    }
    xtx = __riscv_vfmv_f_s_f32m1_f32(x_sum);
    float beta = 2 / xtx;
    return beta;
}

// generate I - beta * v * vT
float* houseHolderHelper(float beta, float* v, int m){
    // printf("HH cycles: %d \n", read_cycles());
    // printf("m: %d \n", m);

    size_t vl;
    uint32_t ind = 0;
    vfloat32m1_t res_vec;
    vfloat32m1_t v_beta_vec;
    // float* v_beta = (float*)calloc(m, sizeof(float)); // Can be malloc, but actually don't need
    // float* result = (float*)calloc(m*m, sizeof(float));
    float* v_beta = (float*)malloc(m* sizeof(float)); // Can be malloc, but actually don't need
    float* result = (float*)malloc(m*m* sizeof(float));
    vector_clear(result, m*m*sizeof(float));
    // printf("HH cycles: %d \n", read_cycles());

    for(int x = m; x > 0; x-=vl) {
        vl = __riscv_vsetvl_e32m1(x);
        v_beta_vec = __riscv_vle32_v_f32m1(v+ind, vl);
        v_beta_vec = __riscv_vfmul_vf_f32m1(v_beta_vec, -beta, vl);
        __riscv_vse32_v_f32m1(v_beta+ind, v_beta_vec, vl);
        ind += vl;
    }
    // printf("HH cycles: %d \n", read_cycles());

    gemm_rvv(result, v, v_beta, m, m, 1); // generate beta*(v x v^T)
    // free(v_beta);

    ptrdiff_t stride = 4*(m+1);

    float* res_ptr = result;
    for(int x = m; x > 0; x-=vl) {
        vl = __riscv_vsetvl_e32m1(x);
        res_vec = __riscv_vlse32_v_f32m1(res_ptr, stride, vl); // Load partial diagonal
        res_vec = __riscv_vfadd_vf_f32m1(res_vec, 1.0, vl); // Subtract from 1 (I - result); CAN REPLACE WITH __riscv_vfnmsub_vf_f32m1
        __riscv_vsse32_v_f32m1(res_ptr, stride, res_vec, vl);  // Store partial diagonal
        res_ptr += vl*m + vl;
    }
    // printf("HH cycles: %d \n", read_cycles());

    return result;
}

// float* houseHolderHelper(float beta, float* v, int m){
//     float* result = (float*)calloc(m*m, sizeof(float));
//     float* identityMatrix = (float*)calloc(m*m, sizeof(float));
//     for (int i = 0; i < m; i++) {
//         identityMatrix[i*m+i] = 1.0;
//     }
//     float* vvT = (float*)calloc(m*m, sizeof(float));
//     gemm_rvv(vvT, v, v, m, m, 1);
//     for (int i = 0; i < m; i++){
//         for (int j = 0; j < m; j++){
//             result[i*m+j] = identityMatrix[i*m+j] - beta * vvT[i*m+j];
//         }
//     }
//     free(identityMatrix);
//     free(vvT);
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
    x[n-1] = y[n-1] / R[(n-1)*n+(n-1)];

    j = 1;
    for (i = n-2; i >= 0; i--) {
        k   = j; // also m-i-1

        // Zero sum vector (splat)
        sum_v = __riscv_vfmv_v_f_f32m1(0.0, 1);
        while (k > 0) {
            vl    = __riscv_vsetvl_e32m1(k);
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
}

void houseHolderQRb(float* A, float* b, int m, int n) {
    float* v; 
    float* x;
    float* H; 
    // float* A_sub = (float*)calloc(m*n, sizeof(float));
    // float* A_sub_updated = (float*)calloc(m*n, sizeof(float));

    float* A_sub = (float*)malloc(m*n* sizeof(float));
    float* A_sub_updated = (float*)malloc(m*n* sizeof(float));
    // vector_clear(A_sub, m*n*sizeof(float));
    // vector_clear(A_sub_updated, m*n*sizeof(float));

    size_t vl;
    uint32_t ind;
    float vtb;
    float beta = 1.0;

    vfloat32m1_t cpy;   // Vector solely for copying
    vfloat32m1_t v_vec;
    vfloat32m1_t b_vec;

    uint32_t vec_size;
    uint32_t mat_size;

    for (int j = 0; j < n; j++){
        // // printf("houseHolderQRb iteration: %d\n", read_cycles());
        // printf("cycles: %d \n", read_cycles());

        // v = (float*)calloc((m-j), sizeof(float));
        // x = (float*)calloc((m-j), sizeof(float));
        v = (float*)malloc((m-j)*sizeof(float));
        x = (float*)malloc((m-j)*sizeof(float));
        vector_clear(v, (m-j)*sizeof(float));
        // vector_clear(x, (m-j)*sizeof(float));

        // Keep for now; can change house to do strided loads, not write x[0] (maintain x)
        for (int i = j; i < m; i++){
            x[i-j] = A[i*n+j];
        }

        beta = house(m-j, x, v);
        H = houseHolderHelper(beta, v, m-j);

        // Compute dot product (v^T,b)
        vtb = dot_product(v, b+j, m-j);

        
        ind = 0;
 
        float coeff = beta*vtb;
        for (int i = m; i > j; i-=vl) {
            vl = __riscv_vsetvl_e32m1(i-j);
            v_vec = __riscv_vle32_v_f32m1(v + ind, vl);
            b_vec = __riscv_vle32_v_f32m1(b + ind + j , vl);
            b_vec = __riscv_vfnmsub_vf_f32m1(v_vec, coeff, b_vec, vl); // b = b - beta*v*v^Tb
            __riscv_vse32_v_f32m1(b + ind + j, b_vec, vl);
            ind += vl;
        }

        // multiply by A submatrix which is m-j * n-j
        // A_sub = (float*)calloc((m-j)*(n-j), sizeof(float));
        // A_sub_updated = (float*)calloc((m-j)*(n-j), sizeof(float));
        // A_sub =         (float*)realloc(A_sub,         (m-j)*(n-j)*sizeof(float));
        // A_sub_updated = (float*)realloc(A_sub_updated, (m-j)*(n-j)*sizeof(float));

        // printf("\tcycles: %d \n", read_cycles());

        for (int i = 0; i < m-j; i++) {
            ind = 0;
            for (int k = n; k > j; k-=vl) {
                vl = __riscv_vsetvl_e32m1(k-j);
                cpy = __riscv_vle32_v_f32m1(A + (i+j)*n + (ind+j), vl);
                __riscv_vse32_v_f32m1(A_sub + i*(n-j) + ind, cpy, vl);
                ind += vl;
            }
        }

        // printf("\tcycles: %d \n", read_cycles());

        gemm_rvv(A_sub_updated, H, A_sub, m-j, n-j, m-j);

        for (int i = 0; i < m-j; i++) {
            ind = 0;
            for (int k = n; k > j; k-=vl) {
                vl = __riscv_vsetvl_e32m1(k-j);
                cpy = __riscv_vle32_v_f32m1(A_sub_updated + i*(n-j) + ind, vl);
                __riscv_vse32_v_f32m1(A + (i+j)*n + (ind+j), cpy, vl);

                ind += vl;
            }
        }

        // free(A_sub);
        // // printf("A_sub free\n");
        // free(A_sub_updated);
        // // printf("A_sub_updated free\n");
        // free(H);
        // // printf("H free\n");
        // free(v);
        // // printf("v free\n");
        // free(x);
        // // printf("x free\n");
    }
}


float* householderQRLS(float* A, float* b, int m, int n){

    // float *x = (float*)calloc(n, sizeof(float));
    float *x = (float*)malloc(n*sizeof(float));
    vector_clear(x, n*sizeof(float));

    houseHolderQRb(A, b, m, n);
    // now b is updated to QTb and A is updated to R
    // solve Rx = Q^Tb
    backSubstitution(A, b, x, m, n);
    return x;
}


// TESTING/DEBUG
// void test_house() {
//     // set test cases here
//     int m = 3;
//     float x[] = {2, 2, 1};
//     float v[] = {0, 0, 0};
//     //
//     float beta = house(m, x, v);
//     for (int i = 0; i < m; i++) {
//         // // printf("x: %f, v: %f\n", x[i], v[i]);
//     }
//     // // printf("beta: %f\n", beta);
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

//     // // printf("H:\n");
//     for (int i = 0; i < m; i++){
//         for (int j = 0; j < m; j++){
//             // // printf("%f ", result[i*m+j]);
//         }
//         // // printf("\n");
//     }
//     float *Hx = calloc(m, sizeof(float));
//     gemm_rvv(Hx, result, x, m, 1, m);
//     // // printf("Hx: \n");
//     for (int i = 0; i < m; i++){
//         // // printf("%f ", Hx[i]);
//     }
//     // // printf("\n");
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
//     // // printf("A:\n");
//     for (int i = 0; i < m; i++){
//         for (int j = 0; j < n; j++){
//             // // printf("%f ", A[i*n+j]);
//         }
//         // // printf("\n");
//     }
//     houseHolderQR(A, m, n);
//     // // printf("R:\n");
//     for (int i = 0; i < m; i++){
//         for (int j = 0; j < n; j++){
//             // // printf("%f ", A[i*n+j]);
//         }
//         // // printf("\n");
//     }
// }

// void test_backSubstitution() {
//     // set test case here
//     int m = 3;
//     int n = 3;
//     float R[] = {1.0f, -2.0f, 1.0f,
//                  0.0f, 1.0f, 6.0f,
//                  0.0f, 0.0f, 1.0f};
//     float y[] = {4.0f, -1.0f, 2.0f};
//     float x[3] = {0.0};

//     // // printf("Calling backsubstition\n");
//     backSubstitution(R, y, x, m, n);

//     // // printf("x: \n");
//     for (int i = 0; i < n; i++) {
//         // // printf("%.3f ", x[i]);
//     }
//     // // printf("\n");

//     // Test case 2
//     m = 10;
//     n = 10;
//     float R1[10][10] = {{0.328406, 0.592807, 0.963605, 0.047914, 0.583902, 0.663277, 0.821321, 0.467992, 0.441245, 0.869632},
//                         {0.000000, 0.287301, 0.810943, 0.594004, 0.299508, 0.506638, 0.629284, 0.176319, 0.036821, 0.382882},
//                         {0.000000, 0.000000, 0.763557, 0.872771, 0.598384, 0.811528, 0.195214, 0.498635, 0.846730, 0.788322},
//                         {0.000000, 0.000000, 0.000000, 0.097213, 0.924739, 0.795865, 0.976783, 0.776233, 0.022980, 0.135772},
//                         {0.000000, 0.000000, 0.000000, 0.000000, 0.793991, 0.748543, 0.093151, 0.347933, 0.043096, 0.423781},
//                         {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.656377, 0.284855, 0.594453, 0.233624, 0.298250},
//                         {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.105906, 0.326819, 0.336461, 0.597929},
//                         {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.722163, 0.119044, 0.746278},
//                         {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.246312, 0.922411},
//                         {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.592532}};
//     float y1[10] = {9, -10,   2,   1, -10,  -3,   4, -15, -15,  -6};
//     float x1[10] = {0.0};
//     float x1_gold[10] = {-1412.0336846940822, -2328.4704431714517, 1889.3309535190674, -1621.1434792606606, 38.486833695540504, -67.49929122866932, 188.0552272912475, -6.519084264460278, -22.977492873375027, -10.126042145412656};

//     // // printf("\nTest Case 2: %dx%d\n", m, n);
//     // // printf("Calling backsubstition\n");
//     backSubstitution(R1[0], y1, x1, m, n);

//     // // printf("x: \n");
//     bool pass = true;
//     for (int i = 0; i < n; i++) {
//         // // printf("%.3f ", x1[i]);
//         pass |= (x1[i]-x1_gold[i]) < 1e-4;
//     }
//     // // printf("\n");
//     // // printf(pass ? "PASS\n" : "FAIL\n");


// }

// void test_houseHolderQRLS() {
//     // int m = 3;
//     // int n = 2;
//     // float A[] = {1.0, -4.0, 2.0, 3.0, 2.0, 2.0}; // Original
//     // float A[] = {2.0, 2.0,
//     //              1.0, -4.0,
//     //              3.0, 2.0};

//     float *x = (float*) calloc(N_DIM, sizeof(float));
//     x = householderQRLS(a_matrix, b_vec, M_DIM, N_DIM);

//     // // printf("ref x: \n");
//     for (int i = 0; i < N_DIM; i++) {
//         // // printf("%f ", x_vec[i]);
//     }
//     // // printf("\n");
//     // // printf("our x: \n");
//     for (int i = 0; i < N_DIM; i++) {
//         // // printf("%f ", x[i]);
//     }
//     // // printf("\n");
// }

// int main() {
//     // test_house();
//     // test_houseHolderHelper();
//     // test_houseHolderQR();
//     // test_backSubstitution();
//     test_houseHolderQRLS();
// }
