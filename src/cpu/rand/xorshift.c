#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>


static inline float rand2float(uint32_t rand_num) {
  uint32_t tmp;
  tmp = (rand_num >> 9) | (0x7F << 23);
  return *((float *)&tmp);
}

uint32_t xorshift() {
  static uint32_t x = 2463534242;
  x=x^(x<<13);
  x=(x>>17);
  x=x^(x<<5);
  return x;
}


uint32_t xorshift_arg(uint32_t x) {
//  static uint32_t x = 2463534242;
  x=x^(x<<13);
  x=(x>>17);
  x=x^(x<<5);
  return x;
}


int main() {
  uint32_t rand;
  for(uint64_t x = 0; x < pow(2, 32); x++) {
    rand = xorshift();
    if((x % (uint32_t) pow(2, 26)) == 0) {
//      printf("\t%d\n", (x % (uint32_t) pow(2, 29)));
      printf("%10d: %10d (%f) -- %10d (%f)\n", x, rand, rand2float(rand), xorshift_arg(rand), rand2float(xorshift_arg(rand)));
    }
  }
  printf("%d\n", rand);

  return 0;
}
