#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "lib.h"
#include "rand.h"
#include "utils.h"
#include "gemm_rvv.h"
#include "riscv_vector.h"

// Include test vector
// #include "small_sqL.h"
// #include "mk12-b2.h"
// #include "sk.h"
// #include "test.h"

// #include "sk_32_fixed.h"
// #include "sk_32_interval.h"
// #include "sk_256_interval.h"
// #include "sk_256_fixed.h"
// #include "sk512_fixed.h"
#include "sk_1024_fixed.h"
// #include "sk_4096_fixed.h"
// #include "sk_8192_fixed.h"


/*
 * Solve a sketch system equstion using Householder (QR Demp)
 * least square solver
 *
 *  A - Input matrix
 *  x - allocated array for solution
 *  b - solution for system
 *  M - # of rows of A
 *  N - # of columns of A
 */
float* pre_sketch_n_solve_rvv(float* S, uint32_t* A_ptr, uint32_t* A_ind, float* A_data, float* b,  uint32_t M, uint32_t N, uint32_t K) {

  // Generate sketched matrix and vector
  uint64_t t0 = read_cycles();
  float* SA = (float*) calloc(K*N, sizeof(float));
  float* Sb = (float*) calloc(N, sizeof(float));
  dsgemm_csc_rvv(SA, S, A_ptr, A_ind, A_data, M, N, M);
  gemm_rvv(Sb, S, b, K, 1, M);
  uint64_t t1 = read_cycles();

  // Solve system
  float* x = householderQRLS(SA, Sb, K, N);
  uint64_t t2 = read_cycles();

  printf("TIME (CYCLES)=============\n");
  printf("  Sketching: %lu\n", t1-t0);
  printf("  Solving: %lu\n", t2-t1);

  return x;
}

// void test_mcycle(uint32_t* arr) {
//   uint32_t t0 = read_cycles();
//   size_t vl = __riscv_vsetvl_e32m1(32);
//   vuint32m8_t x = __riscv_vmv_v_x_u32m8((0x7F << 23), vl); 
//   vuint32m8_t y = __riscv_vle32_v_u32m8(arr, vl);
//   y = __riscv_vadd_vv_u32m8(x, y, vl);
//   __riscv_vse32_v_u32m8(arr, y, vl);
//   uint32_t t1 = read_cycles();

//   printf("TIME (CYCLES)=============\n");
//   printf("  Time0: %d\n", t1-t0);
// }

int main() {
  float* x = (float*) calloc(N_DIM, sizeof(float));
  // static uint32_t arr[32] = {1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10, 1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10, 1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10, 11, 12};
  // test_mcycle(arr);

  x = pre_sketch_n_solve_rvv(sketching_matrix, a_matrix_indptr, a_matrix_indices, a_matrix_data, b_vec, M_DIM, N_DIM, D_DIM);
}
