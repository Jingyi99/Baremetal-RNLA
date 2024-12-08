#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "riscv_vector.h"
#include "rand.h"

#define M 200
#define N 16

void print_vector(char c, uint64_t* v, size_t vl) {
  printf("%c = ", c);
  for(int x=0; x < vl; x++) {
    printf("%d ", v[x]);
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

vuint64m4_t test_vecptr(vuint64m4_t* a, vuint64m4_t* b) {
  size_t vl = __riscv_vsetvl_e64m4(4);
  vuint64m4_t tmp = *b;
  *b = *a;
  *a =  __riscv_vadd_vv_u64m4(*a, tmp, vl);
  return tmp;
}

void run_test_vecptr() {
  uint64_t a[32] = {4, 4, 3, 3, 5, 5, 0, 6, 1, 8, 8, 0, 1, 3, 7, 2, 4, 1, 2, 4, 3, 2, 3, 3, 0, 1, 3, 1, 0, 2, 3, 2};
  uint64_t b[32] = {2, 4, 3, 1, 0, 3, 1, 3, 0, 1, 0, 2, 1, 4, 1, 1, 2, 1, 2, 2, 0, 0, 2, 0, 2, 2, 0, 2, 0, 0, 0, 1};
  uint64_t c[32] = {0.0}; 
  
  size_t vl = __riscv_vsetvl_e64m4(4);
  vuint64m4_t a_v = __riscv_vle64_v_u64m4(a, vl);
  vuint64m4_t b_v = __riscv_vle64_v_u64m4(b, vl);
  vuint64m4_t sum = __riscv_vadd_vv_u64m4(a_v, b_v, vl);
  vuint64m4_t c_v = test_vecptr(&a_v, &b_v);
  __riscv_vse64_v_u64m4(a, a_v, vl);
  __riscv_vse64_v_u64m4(b, b_v, vl);
  __riscv_vse64_v_u64m4(c, c_v, vl);

  print_vector('a', a, vl);
  print_vector('b', b, vl);
  print_vector('c', c, vl);
}

/*
 * Generate -/+1 float from vector of random numbers based upon
 * LSb
 *
 * rand_num - vector of random number generate from PRNG
 * vl - AVL for rand_num
 */
static inline vfloat32m2_t hadamardrize(vuint64m4_t rand_num, size_t vl) {
  // size_t vl = __riscv_vsetvl_e64m4(32);
  vuint64m4_t sign = __riscv_vsll_vx_u64m4(rand_num, 63, vl);
  vuint32m2_t sign_32 = __riscv_vnsrl_wx_u32m2(sign, 32, vl);
  vuint32m2_t res = __riscv_vmv_v_x_u32m2((0x7F << 23), vl);  // Add exponent (127)

  res =  __riscv_vor_vv_u32m2(sign_32, res, vl);
  return __riscv_vreinterpret_v_u32m2_f32m2(res);
}

/* 
 * Generate random float (-1,1) from PRNG with significand being
 * 23 MSbs of upper half of rand_num
 *
 * rand_num - PRNG output (vector reg)
*/
static inline vfloat32m2_t rand2float_64(vuint64m4_t rand_num, size_t vl) {
  vuint64m4_t sign = __riscv_vsll_vx_u64m4(rand_num, 63, vl);
  vuint32m2_t sign_32 = __riscv_vnsrl_wx_u32m2(sign, 32, vl);
  vuint32m2_t rand_num32 = __riscv_vnsrl_wx_u32m2(rand_num, 32, vl);
  vuint32m2_t res = __riscv_vsrl_vx_u32m2(rand_num32, 9, vl);     // Add significand 

  vl = __riscv_vsetvl_e32m2(32);
  res = __riscv_vor_vx_u32m2(res, (0x7F << 23), vl);  // Add exponent (127)

  // Reinterpret as float and change range (-1, 1)
  vfloat32m2_t resf = __riscv_vreinterpret_v_u32m2_f32m2(res);
  resf = __riscv_vfsub_vf_f32m2(resf, 1.0, vl);
  vfloat32m2_t signf = __riscv_vreinterpret_v_u32m2_f32m2(sign_32);
  return __riscv_vfsgnj_vv_f32m2(resf, signf, vl);
}

// Helper function for xoroshiro128p function
static inline vuint64m4_t rotl(const vuint64m4_t x, int k, size_t vl) {
  vuint64m4_t tmp0 = __riscv_vsrl_vx_u64m4(x, 64-k, vl);
  vuint64m4_t tmp1 = __riscv_vsll_vx_u64m4(x, k, vl);
  return __riscv_vor_vv_u64m4(tmp0, tmp1, vl);

}

/* Generates pseudorandom number using xoshiro+
 * 
 * s0 - pointer to first state variable
 * s1 - pointer to second state variable
 * vl - vector length
 *
 * Note: It is anticipated that the state is not global, but passed in
 * current gcc included in chipyard does not support pass by vector 
 * register therefore use pointers
 */
vuint64m4_t xoroshiro128p(vuint64m4_t* s0, vuint64m4_t* s1, size_t vl) {
  vuint64m4_t tmp0;
  vuint64m4_t result = __riscv_vadd_vv_u64m4(*s0, *s1, vl);

  *s1 = __riscv_vxor_vv_u64m4(*s0, *s1, vl);
  *s0 = rotl(*s0, 24, vl);
  *s0 = __riscv_vxor_vv_u64m4(*s0, *s1, vl);
  tmp0 = __riscv_vsll_vx_u64m4(*s1, 16, vl);
  *s0 = __riscv_vxor_vv_u64m4(*s0, tmp0, vl);
  *s1 = rotl(*s1, 37, vl);

  return result;
}

/*
 * Generate a matrix of random floats
 *
 * gen - function to generate numbers
 * rand_mat - pointer to empty matrix 
 */
void genmatrix_xoroshiro128(vfloat32m2_t (*gen)(vuint64m4_t, size_t), float* rand_mat) {
  size_t vl;
  uint64_t seed0[64];
  uint64_t seed1[64];
  
  float S[M][N];  // Sketch matrix
  vl = __riscv_vsetvl_e64m4(32);

  // Load seeds
  vuint64m4_t s0 =__riscv_vle64_v_u64m4(splitmix64_seed0, vl);  
  vuint64m4_t s1 =__riscv_vle64_v_u64m4(splitmix64_seed1, vl); 

  vuint64m4_t rand_v;
  vfloat32m2_t res_float;
  
  // TODO: Expand to matrix sizes not multple of 32 (change VLEN)
  for (uint32_t x = 0; x < M*N; x+=vl) {
    vl = __riscv_vsetvl_e64m4(32);
    rand_v = xoroshiro128p(&s0, &s1, vl);
    res_float = gen(rand_v, vl);
  
    __riscv_vse32_v_f32m2((float *) (rand_mat + x), res_float, vl);
  }

}



int main() {
  uint64_t rand[32] = {0};

  float S[M*N];

  genmatrix_xoroshiro128(rand2float_64, S);
  print_matrix((float *) S, M, N);


  
  // size_t vl = __riscv_vsetvl_e64m4(1);
  // vuint64m4_t s0 = __riscv_vle64_v_u64m4(splitmix64_seed0, vl);
  // vuint64m4_t s1 = __riscv_vle64_v_u64m4(splitmix64_seed1, vl);

  // for(int i=0; i < 1; i++){
  //     vuint64m4_t rand_v = xoshiro128p(&s0, &s1, vl);
  //     __riscv_vse64_v_u64m4(rand, rand_v, vl);
  //     for(int x=0; x < vl; x++) {
  //       printf("0x%lx ", rand[x]);
  //   }
  //   printf("\n");
  // }

  return 0;
}
