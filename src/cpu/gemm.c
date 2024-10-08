// #include <stdio.h>
#include <stdlib.h>

void gemm(float *c_matrix, const float *a_matrix, const float *b_matrix,
            const unsigned int m_dim, const unsigned int n_dim,
            const unsigned int k_dim) {
    for (int i = 0; i < m_dim; i++) {
        for (int j = 0; j < n_dim; j++) {
            for (int k = 0; k < k_dim; k++) {
                c_matrix[i*n_dim + j] += a_matrix[i*k_dim+k] * b_matrix[k*n_dim+j];
            }
        }
    }
}
