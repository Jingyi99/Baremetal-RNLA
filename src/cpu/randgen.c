#include <stdio.h>
#include <stdlib.h>

int randgen() {
   return rand();
}

// dense(left matrix) sparse(right matrix) matmul in CSC format 
void dsgemm_rand_csc(float *c_matrix, const float *a_matrix,
            const int *b_matrix_indptr,
            const int *b_matrix_indices, const float *b_matrix_data,
            const unsigned int m_dim, const unsigned int n_dim,
            const unsigned int k_dim) {
    // int data_ptr = 0;
    for (int col = 0; col < n_dim; col++) {
        int nnz = b_matrix_indptr[col+1] - b_matrix_indptr[col];
        if (nnz > 0) {
            int base_ptr = b_matrix_indptr[col];
            for (int row_ind = 0; row_ind < nnz; row_ind++) {
                for (int i = 0; i < m_dim; i++) {
                    int row = b_matrix_indices[base_ptr + row_ind];
                    c_matrix[i*n_dim+col] += a_matrix[i*k_dim+row] * b_matrix_data[base_ptr+row_ind];
                }
            }
        }
    }
}