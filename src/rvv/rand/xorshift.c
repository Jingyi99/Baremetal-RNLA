#include <stdio.h>
#include <stdint.h>
#include "riscv_vector.h"


// Reinterprets bits as float with significand being
// 23 MSbs of rand_num
static inline vuint32m1_t rand2float(vuint32m1_t rand_num) {
  size_t vl = __riscv_vsetvl_e32m1(8);

  vuint32m1_t res;
  vuint32m1_t exp = __riscv_vmv_v_x_u32m1((0x7F << 23), vl);

  res = __riscv_vsrl_vx_u32m1(rand_num, 9, vl);
  res = __riscv_vor_vv_u32m1(res, exp, vl);

  return res;
}

// Standard XOR PRNG
vuint32m1_t xorshift(vuint32m1_t x) {
  size_t vl = __riscv_vsetvl_e32m1(8);
  vuint32m1_t y;

  y = __riscv_vsll_vx_u32m1(x, 13, vl);
  x = __riscv_vxor_vv_u32m1(x, y, vl);
  x = __riscv_vsrl_vx_u32m1(x, 17, vl);
  y = __riscv_vsll_vx_u32m1(x, 5, vl);
  x = __riscv_vxor_vv_u32m1(x, y, vl);

  return x;
}

int main() {

  uint32_t seeds[8] = {267649, 1018288, 766452, 154837,
                       438273, 694749, 1003440, 89826};
  uint32_t rand[8];
  float    rand_float[8] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};



  size_t vl = __riscv_vsetvl_e32m1(8);
  vuint32m1_t x = __riscv_vle32_v_u32m1(seeds, vl);
  vuint32m1_t res = xorshift(x);
  vuint32m1_t res_float = rand2float(res);

  __riscv_vse32_v_u32m1((uint32_t *) rand, res, vl);
  __riscv_vse32_v_u32m1((uint32_t *) rand_float, res_float, vl);

  for(int x = 0; x < 8; x++) {
    printf("%d (%f)\n", rand[x], rand_float[x]);
    // printf("%f)\n", rand_float[x]);
  }


  return 0;
}
