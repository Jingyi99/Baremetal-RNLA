
/*
 * Perform sketching by outer-product based dense-sparse matrix 
 * multiplcation where the dense sketching matrix is random 
 * matrix generated dynamically.
 * 
 * C - result assumed to be zeroed(!!!)
 * B - 
 */

void randgemm_csc_rvv(  float* C, int* B_cptr, int* B_ind, float* B_data, 
                        uint32_t m_dim, uint32_t n_dim, uint32_t k_dim, ) {
    
    // Random number selected
    size_t vl = __riscv_vsetvl_e32m8(32);

    // Create variables
    uint32_t nnz;
    uint32_t row;
    uint32_t row_ind;
    float* A_ptr;
    float* B_ptr;
    float* C_ptr;

    vfloat32m8_t B_col;
    vfloat32m8_t acc;


    for (int col = 0; col < n_dim; col++) { // Every column of B
        vl = __riscv_vsetvl_e32m8(B_cptr[col+1] - B_cptr[col]); // Number of NNZ in column
        B_col = __riscv_vle32_v_f32m8(B + );    // Load column of B
        for (uint32_t i = 0; i < vl; i++) {
            float element = __riscv_vfmv_f_s_f32m8_f32(B_col);  // Extract next element

            for (uint32_t k = 0; k < k_dim; k++) {  // For every column of sketching matrix
                vfloat32m8_t rand_col = xoroshiro128p();            // Generate partial random column
                acc =__riscv_vfmacc_vf_f32m1(acc, element, a_m_vec, vl);

                // Prep for next iteration
                B_col = __riscv_vfslide1down_vf_f32m8(B_col, 0.0, vl);
            }
        }





// Reference (maybe??)


        if (nnz > 0) {
            int base_ptr = B_cptr[col];
            for (int row_i = 0; row_i < nnz; row_i++) {
                row_ind = base_ptr + row_i;
                row = B_ind[base_ptr + row_i];

                A_ptr = a_matrix + row;


                C_ptr = C+col;

                vuint32m8_t zero32 = __riscv_vmv_s_x_u32m8(0, m_dim);               // Clear accumulator
                vfloat32m8_t acc   = __riscv_vreinterpret_v_u32m8_f32m8(zero32);    // Reinterpret as float
                vfloat32m8_t A_col = __riscv_vls32_v_f32m8(B, vl);                  // Load column of A

                for (uint32_t k = 0; k < count; k++) {  // Iterate across columns of sketching matrix

                }

                for (uint32_t c_m_count = m_dim; c_m_count > 0; c_m_count -=vl) {
                    vl = __riscv_vsetvl_e32m1(c_m_count);
                    vfloat32m1_t acc = __riscv_vlse32_v_f32m1(C_ptr, sizeof(float)*k_dim, vl);
                    vfloat32m1_t a_m_vec = __riscv_vlse32_v_f32m1(A_ptr, sizeof(float) * k_dim, vl);
                    acc = __riscv_vfmacc_vf_f32m1(acc, B_data[row_ind], a_m_vec, vl);
                    __riscv_vsse32_v_f32m1(C_ptr, sizeof(float) * n_dim, acc, vl);
                    A_ptr += vl * k_dim;
                    C_ptr += vl * n_dim;
                }
            }
        }
    }
}