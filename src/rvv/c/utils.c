#include <stdio.h>
#include <stdlib.h>
#include "riscv_vector.h"



void print_vector_uint(uint64_t* v, uint32_t len) {
  // printf("%c = ", c);
  for(int x=0; x < len; x++) {
    printf("%d ", v[x]);
  }
  printf("\n");
}

void print_vector(float* v, uint32_t len) {
  // printf("%c = ", c);
  for(int x=0; x < len; x++) {
    printf("%f ", v[x]);
  }
  printf("\n");
}

void print_matrix(float* A, int r, int c) {
  uint32_t i, j;
  for(i = 0; i < r; i++) {
    for(j = 0; j < c-1; j++) {
      printf("%5.2f ", *(A + i*c + j));
    }
    printf("%5.2f\n", *(A + i*c + j));
  }
}

void* allocate_vector_clear(uint32_t bytes) {
    void* arr = malloc(bytes);
    size_t vl = __riscv_vsetvl_e8m8(bytes); // Just to ensure VLMAX
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
    size_t vl = __riscv_vsetvl_e8m8(bytes); // Just to ensure VLMAX    
    vuint8m8_t zero = __riscv_vmv_v_x_u8m8(0, vl);
    void* base = arr;
    for (int b=bytes; b > 0; b-=vl) {
        vl = __riscv_vsetvl_e8m8(b); // Just to ensure VLMAX
        __riscv_vse8_v_u8m8(base, zero, vl);
        base += vl;
    }
}


// static inline float dot_product(float* x, float* y, int m) {
//     size_t vl;
//     int32_t j = 0;
//     vfloat32m1_t x_p;
//     vfloat32m1_t y_p;
//     vfloat32m1_t prod;
//     vfloat32m1_t sum = __riscv_vfmv_v_f_f32m1(0.0, 1);

//     for (int i = m; i > 0; i-=vl){
//         vl = __riscv_vsetvl_e32m1(i);
//         x_p = __riscv_vle32_v_f32m1(x + j, vl);
//         y_p = __riscv_vle32_v_f32m1(y + j, vl);
//         prod = __riscv_vfmul_vv_f32m1(x_p, y_p, vl);
//         sum = __riscv_vfredosum_vs_f32m1_f32m1(prod, sum, vl);
//         j += vl;
//     }

//     return __riscv_vfmv_f_s_f32m1_f32(sum);
// }

