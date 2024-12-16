#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "rand.h"
#include "utils.h"
#include "gemm_rvv.h"
#include "riscv_vector.h"

// Include test vector
// #include "small_sqL.h"
#include "../../dataset/sns/sk_32_fixed.h"

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
float* sketch_n_solve_rvv(uint32_t* A_ptr, uint32_t* A_ind, float* A_data, float* b, uint32_t M, uint32_t N, uint32_t K) {
  // Generate sketch matrix
  float* S = (float*) calloc(K*M, sizeof(float));
  genmatrix_xorshift(xorshift, rand2float_32, S, M, M);

  // Generate sketched matrix and vector
  float* SA = (float*) calloc(K*N, sizeof(float));
  float* Sb = (float*) calloc(N, sizeof(float));
  dsgemm_csc_rvv(SA, S, A_ptr, A_ind, A_data, M, N, M); 
  gemm_rvv(Sb, S, b, K, 1, M);

  // Solve system
  return householderQRLS(SA, Sb, M, N);
}

int main() {
  float* x = (float*) calloc(N_DIM, sizeof(float));
  // float* b = (float*) calloc(M_DIM, sizeof(float));

  // // Copy single column from solution matrix (not always needed)
  // for(int i = 0; i < M_DIM; i++) {
  //   b[i] = B[i][0];
  // }

  x = sketch_n_solve_rvv(a_matrix_indptr, a_matrix_indices, a_matrix_data, b_vec, M_DIM, N_DIM, 3*N_DIM);
}