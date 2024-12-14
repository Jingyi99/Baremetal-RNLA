#pragma once 
#ifndef UTILS_HEADER
#define UTILS_HEADER
#include <stdlib.h>
#include "riscv_vector.h"

// Function to print a vector of unsigned integers (uint64_t)
void print_vector_uint(uint64_t* v, uint32_t len);

// Function to print a vector of floats
void print_vector(float* v, uint32_t len);

// Function to print a matrix of floats (row-major format)
void print_matrix(float* A, int r, int c);

// Function to allocate and clear memory for a vector
void* allocate_vector_clear(uint32_t bytes);

// Function to clear the contents of a vector
void vector_clear(void* arr, uint32_t bytes);

// Inline function to compute the dot product of two vectors
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

#endif
