#pragma once 

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
      printf("%.2f ", *(A + i*c + j));
    }
    printf("%.2f\n", *(A + i*c + j));
  }
}
