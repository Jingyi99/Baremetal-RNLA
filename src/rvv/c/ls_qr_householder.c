#include <lib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "utils.h"
#include "gemm_rvv.h"
#include "householder.h"
#include "riscv_vector.h"

// textbook implementation
float house(int m, float* x, float* v) {
    float beta = 0.0;
    float sigma = dot_product(x+1, x+1, m-1);

    // for (int i = 1; i < m; i++) {
    //     sigma += x[i] * x[i];
    // }

    // memcpy(v, x, m * sizeof(float));

    uint32_t ind = 0;
    vfloat32m1_t cpy;
    size_t vl;
    float* ptr = x;  // Just to get rid of pointer arithmetic warning
    for (uint32_t x = m; x > 0; x -= vl) {
        ptr += ind;
        vl = __riscv_vsetvl_e32m1(x);
        cpy = __riscv_vle32_v_f32m1(ptr, vl);
        __riscv_vse32_v_f32m1(v + ind, cpy, vl);
        ind += vl;
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
            // for (int i = 0; i < m; i++) {
            //     v[i] = v[i]/v0;
            // }
            uint32_t ind = 0;
            vfloat32m1_t tmp;
            for (int i = m; i > 0 ; i-=vl) {
                vl = __riscv_vsetvl_e32m1(i);
                tmp = __riscv_vle32_v_f32m1(v+ind, vl);
                tmp = __riscv_vfdiv_vf_f32m1(tmp, v0, vl);
                __riscv_vse32_v_f32m1(v+ind, tmp, vl);
                ind += vl;
            }

        } else {
            // for (int i = 0; i < m; i++) {
            //     v[i] = 0;
            // }
            vector_clear(v, m*sizeof(float));
        }
    }
    return beta;
}


// // more intuitive implementation
// float house(int m, float* x, float* v){
//     float xNorm = 0;
//     size_t vl;
//     int32_t i;
//     int32_t j = 0;

//     vfloat32m1_t x_p;
//     vfloat32m1_t x_sum;
//     vfloat32m1_t x_div;
//     vfloat32m1_t x_prod;

//     x_sum = __riscv_vfmv_v_f_f32m1(0.0, 1);
//     for (i = m; i > 0; i-=vl){
//         vl = __riscv_vsetvl_e32m1(i);
//         x_p = __riscv_vle32_v_f32m1(x + j, vl);
//         x_prod = __riscv_vfmul_vv_f32m1(x_p, x_p, vl);
//         x_sum = __riscv_vfredosum_vs_f32m1_f32m1(x_prod, x_sum, vl);
//         j += vl;
//     }

//     xNorm = dot_product(x, x, m);
//     xNorm = sqrt(xNorm);
//     float sign = (x[0] >= 0) ? 1 : -1;

//     x[0] += (x[0] >= 0) ? xNorm : -xNorm; // x[0] + sign * xNorm;
//     float xtx = 0;
//     float x0 = x[0];

//     j = 0;
//     x_sum = __riscv_vfmv_v_f_f32m1(0.0, 1);
//     for (i = m; i > 0; i-=vl){
//         vl = __riscv_vsetvl_e32m1(i);
//         x_p = __riscv_vle32_v_f32m1(x + j, vl);
//         x_div = __riscv_vfdiv_vf_f32m1(x_p, x0, vl);
//         __riscv_vse32_v_f32m1(v + j, x_div, vl);
//         x_prod = __riscv_vfmul_vv_f32m1(x_div, x_div, vl);
//         x_sum = __riscv_vfredosum_vs_f32m1_f32m1(x_prod, x_sum, vl);
//         j += vl;
//     }

//     xtx = __riscv_vfmv_f_s_f32m1_f32(x_sum);
//     printf("\nxTx:%f\n", xtx);

//     float beta = 2 / xtx;
//     return beta;
// }

// generate I - beta * v * vT
float* houseHolderHelper(float beta, float* v, int m){
    size_t vl;
    uint32_t ind = 0;
    vfloat32m1_t res_vec;
    vfloat32m1_t v_beta_vec;
    float* v_beta = (float*)malloc(m* sizeof(float)); // Can be malloc, but actually don't need
    float* result = (float*)malloc(m*m* sizeof(float));
    vector_clear(result, m*m*sizeof(float));
    // vector_clear(v_beta, m*sizeof(float));

    for(int x = m; x > 0; x-=vl) {
        vl = __riscv_vsetvl_e32m1(x);
        v_beta_vec = __riscv_vle32_v_f32m1(v+ind, vl);
        v_beta_vec = __riscv_vfmul_vf_f32m1(v_beta_vec, -beta, vl);
        __riscv_vse32_v_f32m1(v_beta+ind, v_beta_vec, vl);
        ind += vl;
    }

    // Perform cross product
    gemm_rvv(result, v, v_beta, m, m, 1); // generate beta*(v x v^T)
    // free(v_beta);

    const ptrdiff_t stride = 4*(m+1);

    float* res_ptr = result;
    for(int x = m; x > 0; x-=vl) {
        vl = __riscv_vsetvl_e32m1(x);
        res_vec = __riscv_vlse32_v_f32m1(res_ptr, stride, vl); // Load partial diagonal
        res_vec = __riscv_vfadd_vf_f32m1(res_vec, 1.0, vl); // Subtract from 1 (I - result); CAN REPLACE WITH __riscv_vfnmsub_vf_f32m1
        __riscv_vsse32_v_f32m1(res_ptr, stride, res_vec, vl);  // Store partial diagonal
        res_ptr += vl*m + vl;
    }

    return result;
}



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
        x[i] = R[i*n+i] ? (y[i] - sum) / R[i*n+i] : 0.0; // Prevent divide by zero
        j++;
    }
}

void houseHolderQRb(float* A, float* b, int m, int n) {
    float* v; 
    float* x;
    float* H; 
    float* A_sub = (float*)malloc(m*n* sizeof(float));
    float* A_sub_updated = (float*)malloc(m*n* sizeof(float));
    // vector_clear(A_sub, m*n*sizeof(float));
    // vector_clear(A_sub_updated, m*n*sizeof(float));

    float vtb;
    float coeff;
    float beta = 1.0;

    size_t vl;
    uint32_t ind;
    uint32_t vec_size;
    uint32_t mat_size;

    vfloat32m1_t cpy;   // Vector solely for copying
    vfloat32m1_t v_vec;
    vfloat32m1_t b_vec;

    for (int j = 0; j < n; j++){
        v = (float*)malloc((m-j)*sizeof(float));
        x = (float*)malloc((m-j)*sizeof(float));
        vector_clear(v, (m-j)*sizeof(float));
        // vector_clear(x, (m-j)*sizeof(float));

        // Keep for now; can change house to do strided loads, not write x[0] (maintain x)
        for (int i = j; i < m; i++){
            x[i-j] = A[i*n+j];
        }

        beta = house(m-j, x, v);
        printf("\nbeta:%f\n", beta);
        H = houseHolderHelper(beta, v, m-j);

        // Compute dot product (v^T,b)
        vtb = dot_product(v, b+j, m-j);

        ind = 0;
        coeff = beta*vtb;
        for (int i = m; i > j; i-=vl) {
            vl = __riscv_vsetvl_e32m1(i-j);
            v_vec = __riscv_vle32_v_f32m1(v + ind, vl);
            b_vec = __riscv_vle32_v_f32m1(b + ind + j , vl);
            b_vec = __riscv_vfnmsub_vf_f32m1(v_vec, coeff, b_vec, vl); // b = b - beta*v*v^Tb
            __riscv_vse32_v_f32m1(b + ind + j, b_vec, vl);
            ind += vl;
        }

        // multiply by A submatrix which is m-j * n-j
        // A_sub =         (float*)realloc(A_sub,         (m-j)*(n-j)*sizeof(float));
        // A_sub_updated = (float*)realloc(A_sub_updated, (m-j)*(n-j)*sizeof(float));


        for (int i = 0; i < m-j; i++) {
            ind = 0;
            for (int k = n; k > j; k-=vl) {
                vl = __riscv_vsetvl_e32m1(k-j);
                cpy = __riscv_vle32_v_f32m1(A + (i+j)*n + (ind+j), vl);
                __riscv_vse32_v_f32m1(A_sub + i*(n-j) + ind, cpy, vl);
                ind += vl;
            }
        }

        vector_clear(A_sub_updated, (m-j)*(n-j)*sizeof(float));
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
        free(H);
        // // printf("H free\n");
        free(v);
        // // printf("v free\n");
        free(x);
        // // printf("x free\n");
    }
}


float* householderQRLS(float* A, float* b, int m, int n){

    float *x = (float*)malloc(n*sizeof(float));
    vector_clear(x, n*sizeof(float));

    houseHolderQRb(A, b, m, n);

    print_matrix(A, m, n);
    printf("\nx:\n");
    print_vector(b, n);

    // now b is updated to QTb and A is updated to R
    // solve Rx = Q^Tb
    backSubstitution(A, b, x, m, n);
    return x;
}



