#include <riscv_vector.h>

void matmul_rvv(float *c_matrix, const float *a_matrix, const float *b_matrix,
            const unsigned int m_dim, const unsigned int n_dim,
            const unsigned int k_dim) {
    size_t vl;
    for (int i = 0; i < m_dim; i++) {
        const float *b_n_ptr = b_matrix;
        float *c_n_ptr = c_matrix;
        for (size_t c_n_count = n_dim; c_n_count; c_n_count -= vl) {
            vl = __riscv_vsetvl_e32m1(c_n_count);
            const float *a_k_ptr = a_matrix;
            const float *b_k_ptr = b_n_ptr;
            vfloat32m1_t acc = __riscv_vle32_v_f32m1(c_n_ptr, vl);
            for (size_t k = 0; k < k_dim; ++k) {
                vfloat32m1_t b_n_data = __riscv_vle32_v_f32m1(b_k_ptr, vl);
                acc = __riscv_vfmacc_vf_f32m1(acc, *a_k_ptr, b_n_data, vl);
                b_k_ptr +=  n_dim;
                a_k_ptr++;
            }
            __riscv_vse32_v_f32m1(c_n_ptr, acc, vl);
            c_n_ptr += vl;
            b_n_ptr += vl;
        }
        a_matrix += k_dim;
        c_matrix += n_dim;
    }
}
