#include <stdint.h>
#include <stdio.h>
#include "lapacke.h"
#include "lapacke_utils.h"

#define n 5

void print_matrix(float* A, int r, int c) {
  uint32_t i, j;
  for(i = 0; i < r; i++) {
    for(j = 0; j < c-1; j++) {
      printf("%5.8f ", *(A + i*c + j));
    }
    printf("%5.8f\n", *(A + i*c + j));
  }
}

int main() {
  // int n = 5;
  float A[n*n] = { 1.52282485, -1.33105524, -0.46328294, -0.66219897, -1.35410243,
       -1.33105524,  2.29722519,  0.56088177,  0.84563591,  1.69904816,
       -0.46328294,  0.56088177,  0.96620711,  0.20347949,  0.37860907,
       -0.66219897,  0.84563591,  0.20347949,  0.59777389,  0.83557352,
       -1.35410243,  1.69904816,  0.37860907,  0.83557352,  2.24178834};

  int ret = LAPACKE_spotrf2(LAPACK_ROW_MAJOR, 'L', n, A, n);

  if(ret==0) {
    printf("Cholesky SUCCESSFUL\n");
  } else {
    printf("Cholesky FAILED\n");
  }

  printf("%d\n", A);
  print_matrix(A, n, n);

  return ret;
}
