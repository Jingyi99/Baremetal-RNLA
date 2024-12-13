#include <riscv_vector.h>
#include "gemm_rvv.h"

void dsgemm_csc_rvv(float *c_matrix, const float *a_matrix, 
                        const int *b_matrix_indptr, const int *b_matrix_indices,
                        const float *b_matrix_data, const unsigned int m_dim,
                        const unsigned int n_dim, const unsigned int k_dim) {
    size_t vl;
    for (int col = 0; col < n_dim; col++) {
        int nnz = b_matrix_indptr[col+1] - b_matrix_indptr[col];
        if (nnz > 0) {
            int base_ptr = b_matrix_indptr[col];
            for (int row_i = 0; row_i < nnz; row_i++) {
                int row_ind = base_ptr + row_i;
                int row = b_matrix_indices[base_ptr + row_i];
                const float *a_m_ptr = a_matrix + row;
                float *c_m_ptr = c_matrix+col;
                for (size_t c_m_count = m_dim; c_m_count; c_m_count -=vl) {
                    vl = __riscv_vsetvl_e32m1(c_m_count);
                    vfloat32m1_t acc = __riscv_vlse32_v_f32m1(c_m_ptr, sizeof(float)* k_dim, vl);
                    vfloat32m1_t a_m_vec = __riscv_vlse32_v_f32m1(a_m_ptr, sizeof(float) * k_dim, vl);
                    acc = __riscv_vfmacc_vf_f32m1(acc, b_matrix_data[row_ind], a_m_vec, vl);
                    __riscv_vsse32_v_f32m1(c_m_ptr, sizeof(float) * n_dim, acc, vl);
                    a_m_ptr += vl * k_dim;
                    c_m_ptr += vl * n_dim;
                }
            }
        }
    }
}

