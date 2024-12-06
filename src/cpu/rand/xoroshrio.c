#include <stdio.h>
#include <stdint.h>

uint32_t rol64(uint32_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

typedef struct  {
  uint32_t s[4];
} xoshiro128p_state;

static inline float rand2float(uint32_t rand_num) {
  uint32_t tmp;
  tmp = (rand_num >> 9) | (0x7F << 23);
  return *((float *)&tmp);
}

uint32_t xoshiro128p(xoshiro128p_state *state) {
  uint32_t* s = state->s;
  uint32_t const result = s[0] + s[3];
  uint32_t const t = s[1] << 17;

  s[2] ^= s[0];
  s[3] ^= s[1];
  s[1] ^= s[2];
  s[0] ^= s[3];

  s[2] ^= t;
  s[3] = rol64(s[3], 45);

  return result;
}

int main() {

  int print_cycle = 1e4;
  uint32_t rand_num = 0;
  xoshiro128p_state seed;
  seed.s[0] = 0x4C32F8B9;
  seed.s[1] = 0xFA2F8D6F;
  seed.s[2] = 0xF1436CEF;
  seed.s[3] = 0xEFAB4D68;

  for(int x=0; x < 1e6; x++) {
    if ((x+1) % print_cycle == 1) {
      rand_num = xoshiro128p(&seed);
      printf("n=%d: %f (%x)[%x]\n", x, rand2float(rand_num), rand_num, rand2float(rand_num));
    }
  }

  return 0;
}
