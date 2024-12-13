#include <stdint.h>
#include <stddef.h>

// returns nonzero if errors
//int verify_matrix(float *result, float *gold, size_t m_dim, size_t n_dim) {
//    float tolerance = 1e-2;
//    for (uint64_t i = 0; i < m_dim; ++i) {
//        for (uint64_t j = 0; j < n_dim; ++j) {
//            uint64_t idx = i * n_dim + j;
//            // printf("i: %u, j: %u, idx: %u, result: %f\n", i, j, idx, tolerance);
//            if (fabs(result[idx]-gold[idx]) > tolerance){
//                printf("got: %d, expect: %d, index: %u\n", (int) result[idx], (int) gold[idx], (int) idx);
//                return (i+j == 0? -1 : idx);
//            }
//        }
//    }
//    return 0;
//}

/* ================ RISC-V specific definitions ================ */
#define READ_CSR(REG) ({                          \
  unsigned long __tmp;                            \
  asm volatile ("csrr %0, " REG : "=r"(__tmp));  \
  __tmp; })

#define WRITE_CSR(REG, VAL) ({                    \
  asm volatile ("csrw " REG ", %0" :: "rK"(VAL)); })

#define SWAP_CSR(REG, VAL) ({                     \
  unsigned long __tmp;                            \
  asm volatile ("csrrw %0, " REG ", %1" : "=r"(__tmp) : "rK"(VAL)); \
  __tmp; })

#define SET_CSR_BITS(REG, BIT) ({                 \
  unsigned long __tmp;                            \
  asm volatile ("csrrs %0, " REG ", %1" : "=r"(__tmp) : "rK"(BIT)); \
  __tmp; })

#define CLEAR_CSR_BITS(REG, BIT) ({               \
  unsigned long __tmp;                            \
  asm volatile ("csrrc %0, " REG ", %1" : "=r"(__tmp) : "rK"(BIT)); \
  __tmp; })


static size_t read_cycles() {
  #if defined(RISCV)
    return READ_CSR("mcycle");
  #endif
}
