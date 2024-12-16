#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "cholesky.h"
#include "riscv_vector.h"

// This is destructive; Writes L into A
void cholesky(float* A, int n) {
  // size_t vl;
  // int len;
  // float l_jj;
  // float l_ij;
  float inner_prod;
  for (uint32_t j = 0; j < n; j++) {
    // Find vector length
    
    // Call dot_product
    inner_prod = dot_product(A + j*n, A + j*n, j);
    // Sunbract
    // l_jj = sqrt(A[j*n + j] - inner_prod);
    A[j*n+j] = sqrt(A[j*n + j] - inner_prod);
    
    for (uint32_t i = j+1; i < n; i++) {
      inner_prod = dot_product(A + i*n, A + j*n, j);
      // l_ij = ((A[i*n + j]) - inner_prod)/l_jj;
      A[i*n+j] = ((A[i*n + j]) - inner_prod)/A[j*n+j];
    }
  }
}

