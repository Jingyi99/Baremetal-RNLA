#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "rand.h"
#include "utils.h"
#include "riscv_vector.h"

// #ifdef DEBUG
//   void basic_test(uint32_t seeds[]) {
//     uint32_t rand[64] = {0};
//     float    rand_float[64] = {0.0};

//     size_t vl = __riscv_vsetvl_e32m8(32);
//     vuint32m8_t x0 = __riscv_vle32_v_u32m8(seeds, vl);
//     vuint32m8_t x1 = __riscv_vle32_v_u32m8(seeds+32, vl);
//     vuint32m8_t res0 = xorshift(x0);
//     vuint32m8_t res1 = xorshift(x1);
//     vuint32m8_t res_float0 = rand2float_32(res0);
//     vuint32m8_t res_float1 = rand2float_32(res1);

//     __riscv_vse32_v_u32m8((uint32_t *) rand, res0, vl);
//     __riscv_vse32_v_u32m8((uint32_t *) rand+32, res1, vl);
//     __riscv_vse32_v_u32m8((uint32_t *) rand_float, res_float0, vl);
//     __riscv_vse32_v_u32m8((uint32_t *) rand_float+32, res_float1, vl);

//     for(int x = 0; x < 64; x++) {
//       printf("%d (%f)\n", rand[x], rand_float[x]);
//     }
//   }
// #endif

/* 
 * Generate random float (-1,1) from PRNG with significand being
 * 23 Mrand_matbs of rand_num
 *
 * rand_num - output from PRNG
 */
vfloat32m8_t rand2float_32(vuint32m8_t rand_num) {
  size_t vl = __riscv_vsetvl_e32m8(32);

  vuint32m8_t res;
  vfloat32m8_t resf;
  vfloat32m8_t signf;
  vuint32m8_t exp = __riscv_vmv_v_x_u32m8((0x7F << 23), vl);
  vuint32m8_t sign = __riscv_vsll_vx_u32m8(rand_num, 31, vl); // rand_matign based on even/odd

  res = __riscv_vsrl_vx_u32m8(rand_num, 9, vl);
  res = __riscv_vor_vv_u32m8(res, exp, vl);
  res = __riscv_vor_vv_u32m8(res, sign, vl);

  resf = __riscv_vreinterpret_v_u32m8_f32m8(res);
  resf = __riscv_vfsub_vf_f32m8(resf, 1.0, vl);
  signf = __riscv_vreinterpret_v_u32m8_f32m8(sign);

  return __riscv_vfsgnj_vv_f32m8(resf, signf, vl);
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

/*
 * Generate -/+1 float from vector of random numbers based upon
 * Lrand_matb
 *
 * rand_num - vector of random number generate from PRNG
 * vl - AVL for rand_num
 */
vfloat32m8_t hadamardrize_e32(vuint32m8_t rand_num, size_t vl) {
  vuint32m8_t sign = __riscv_vsll_vx_u32m8(rand_num, 31, vl); // rand_mathift Lrand_matb to Mrand_matb
  vuint32m8_t res  = __riscv_vmv_v_x_u32m8((0x7F << 23), vl);  // Add exponent (127)
  res = __riscv_vor_vv_u32m8(sign, res, vl);
  return __riscv_vreinterpret_v_u32m8_f32m8(res);
}



// x - beginning x coordinate of column
// y - beginning y coordinate of column
// vl - vector length
vuint32m8_t gen_by_coordinates32(uint32_t x, uint32_t y, size_t vl) {
  vuint32m8_t ind = __riscv_vid_v_u32m8(vl);
  vuint32m8_t seeds = __riscv_vadd_vx_u32m8(ind, x, vl);
  vuint32m8_t y_vec = __riscv_vadd_vx_u32m8(ind, x, vl);
  seeds = __riscv_vsll_vx_u32m8(seeds, 16, vl);
  seeds = __riscv_vsll_vx_u32m8(seeds, y, vl);
  y_vec = xorshift(seeds);
  return y_vec;

  // ind = iota 
  // x_ind = ind + x
  // y_ind = ind + x
  // x_ind = x_ind << 32 | y_ind
  // Call PRNG with seeds
}

/*
 * Generate MxN random matrix using Xorshift PRNG function
 *
 * gen - PRNG function (galois or xorshift)
 * conv - conversion function (float or hadamard)
 * rand_mat - allocated array for generated matrix
 * M - # of rows of rand_mat
 * N - # of columns of rand_mat
 */
void genmatrix_xorshift(vuint32m8_t (*gen)(vuint32m8_t), vfloat32m8_t (*conv)(vuint32m8_t), float* rand_mat, uint32_t M, uint32_t N) {
  size_t vl = __riscv_vsetvl_e32m8(32);

  // Load seed values to vector registers
  vuint32m8_t x0 = __riscv_vle32_v_u32m8(xor_seeds, vl);
  vuint32m8_t x1 = __riscv_vle32_v_u32m8(xor_seeds+32, vl);

  // TODO: Expand to matrix sizes not multple of 32 (change VLEN)
  for (uint32_t x = 0; x < ceil(M*N/64); x++) {
    x0 = gen(x0);
    x1 = gen(x1);
    vfloat32m8_t res_float0 = conv(x0);
    vfloat32m8_t res_float1 = conv(x1);
  
    // __riscv_vse32_v_u32m8(rand + 64*x, x0, vl);
    // __riscv_vse32_v_u32m8(rand+32 + 64*x, x1, vl);
    __riscv_vse32_v_f32m8(rand_mat + 64*x, res_float0, vl);
    __riscv_vse32_v_f32m8(rand_mat+32 + 64*x, res_float1, vl);
  }
}

// int main() {
//   // uint32_t rand[M][N] = {0};

//   // size_t vl = __riscv_vsetvl_e32m8(32);
//   // // printf("vl: %d\n", vl);

//   // // basic_test(seeds);

//   // float rand_mat[M][N];  // rand_matketch matrix

//   // // Load seed values to vector registers
//   // vuint32m8_t x0 = __riscv_vle32_v_u32m8(seeds, vl);
//   // vuint32m8_t x1 = __riscv_vle32_v_u32m8(seeds+32, vl);

//   // // TODO: Expand to matrix sizes not multple of 32 (change VLEN)
//   // for (uint32_t x = 0; x < ceil(M*N/64); x++) {
//   //   x0 = xorshift(x0);
//   //   x1 = xorshift(x1);
//   //   vfloat32m8_t res_float0 = rand2float_32(x0);
//   //   vfloat32m8_t res_float1 = rand2float_32(x1);
  
//   //   __riscv_vse32_v_u32m8((uint32_t *) &rand[0] + 64*x, x0, vl);
//   //   __riscv_vse32_v_u32m8((uint32_t *) &rand[0]+32 + 64*x, x1, vl);
//   //   __riscv_vse32_v_u32m8((uint32_t *) &rand_mat[0] + 64*x, res_float0, vl);
//   //   __riscv_vse32_v_u32m8((uint32_t *) &rand_mat[0]+32 + 64*x, res_float1, vl);
//   // }

//   // print_matrix((float *) &rand_mat[0], M, N);

//   return 0;
// }
