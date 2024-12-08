#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "riscv_vector.h"
#include "rand.h"

#define M 20
#define N 48

void print_matrix(float* A, int r, int c) {
  uint32_t i, j;
  for(i = 0; i < r; i++) {
    for(j = 0; j < c-1; j++) {
      printf("%f ", *(A + i*c + j));
    }
    printf("%f\n", *(A + i*c + j));
  }
}



// Reinterprets bits as float with significand being
// 23 MSbs of rand_num
static inline vuint32m8_t rand2float_32(vuint32m8_t rand_num) {
  size_t vl = __riscv_vsetvl_e32m8(32);

  vuint32m8_t res;
  vuint32m8_t exp = __riscv_vmv_v_x_u32m8((0x7F << 23), vl);
  vuint32m8_t sign = __riscv_vsll_vx_u32m8(rand_num, 31, vl); // Sign based on even/odd

  res = __riscv_vsrl_vx_u32m8(rand_num, 9, vl);
  res = __riscv_vor_vv_u32m8(res, exp, vl);
  res = __riscv_vor_vv_u32m8(res, sign, vl);

  return res;
}

// Standard XOR PRNG
vuint32m8_t xorshift(vuint32m8_t x) {
  size_t vl = __riscv_vsetvl_e32m8(32);
  vuint32m8_t y;

  y = __riscv_vsll_vx_u32m8(x, 13, vl);
  x = __riscv_vxor_vv_u32m8(x, y, vl);
  x = __riscv_vsrl_vx_u32m8(x, 17, vl);
  y = __riscv_vsll_vx_u32m8(x, 5, vl);
  x = __riscv_vxor_vv_u32m8(x, y, vl);

  return x;
}

// Perfoms Galois LFSR
vuint32m8_t galois(vuint32m8_t x) {
  size_t vl = __riscv_vsetvl_e32m8(32);
  vuint32m8_t lsb = __riscv_vand_vx_u32m8(x, 0x1, vl);
  vbool4_t lsb_bool = __riscv_vmseq_vx_u32m8_b4(lsb, 0x1, vl);
  x = __riscv_vsrl_vx_u32m8(x, 1, vl);
  x = __riscv_vxor_vx_u32m8_m(lsb_bool, x, 0x80017BAE, vl);
  return x;
}


void basic_test(uint32_t seeds[]) {
  uint32_t rand[64] = {0};
  float    rand_float[64] = {0.0};

  size_t vl = __riscv_vsetvl_e32m8(32);
  vuint32m8_t x0 = __riscv_vle32_v_u32m8(seeds, vl);
  vuint32m8_t x1 = __riscv_vle32_v_u32m8(seeds+32, vl);
  vuint32m8_t res0 = xorshift(x0);
  vuint32m8_t res1 = xorshift(x1);
  vuint32m8_t res_float0 = rand2float_32(res0);
  vuint32m8_t res_float1 = rand2float_32(res1);

  __riscv_vse32_v_u32m8((uint32_t *) rand, res0, vl);
  __riscv_vse32_v_u32m8((uint32_t *) rand+32, res1, vl);
  __riscv_vse32_v_u32m8((uint32_t *) rand_float, res_float0, vl);
  __riscv_vse32_v_u32m8((uint32_t *) rand_float+32, res_float1, vl);

  for(int x = 0; x < 64; x++) {
    printf("%d (%f)\n", rand[x], rand_float[x]);
  }
}


int main() {
  uint32_t rand[M][N] = {0};

  size_t vl = __riscv_vsetvl_e32m8(32);
  printf("vl: %d\n", vl);

  // basic_test(seeds);

  float S[M][N];  // Sketch matrix

  // Load seed values to vector registers
  vuint32m8_t x0 = __riscv_vle32_v_u32m8(seeds, vl);
  vuint32m8_t x1 = __riscv_vle32_v_u32m8(seeds+32, vl);

  // TODO: Expand to matrix sizes not multple of 32 (change VLEN)
  for (uint32_t x = 0; x < ceil(M*N/64); x++) {
    x0 = xorshift(x0);
    x1 = xorshift(x1);
    vuint32m8_t res_float0 = rand2float_32(x0);
    vuint32m8_t res_float1 = rand2float_32(x1);
  
    __riscv_vse32_v_u32m8((uint32_t *) &rand[0] + 64*x, x0, vl);
    __riscv_vse32_v_u32m8((uint32_t *) &rand[0]+32 + 64*x, x1, vl);
    __riscv_vse32_v_u32m8((uint32_t *) &S[0] + 64*x, res_float0, vl);
    __riscv_vse32_v_u32m8((uint32_t *) &S[0]+32 + 64*x, res_float1, vl);
  }

  print_matrix((float *) &S[0], M, N);

  return 0;
}
