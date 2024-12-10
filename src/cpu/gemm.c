#include <stdlib.h>

void gemm(double *c_matrix, const double *a_matrix, const double *b_matrix,
            const unsigned int m_dim, const unsigned int n_dim,
            const unsigned int k_dim) {
    // assuming that c_matrix is zero, quick fix
    // for (int i = 0; i < m_dim; i++) {
    //     for (int j = 0; j < n_dim; j++) {
    //         for (int k = 0; k < k_dim; k++) {
    //             c_matrix[i*n_dim + j] =0.0;
    //         }
    //     }
    // }
    for (int i = 0; i < m_dim; i++) {
        for (int j = 0; j < n_dim; j++) {
            for (int k = 0; k < k_dim; k++) {
                c_matrix[i*n_dim + j] += a_matrix[i*k_dim+k] * b_matrix[k*n_dim+j];
            }
        }
    }
}

void blocked_gemm(double *c_matrix, const double *a_matrix, const double *b_matrix,
            const unsigned int m_dim, const unsigned int n_dim,
            const unsigned int k_dim, const unsigned int bm_dim,
            const unsigned int bk_dim, const unsigned int bn_dim) {
    for (int i = 0; i < m_dim; i += bm_dim) {
        for (int j = 0; j < n_dim; j += bn_dim) {
            for (int k = 0; k < k_dim; k += bk_dim) {
                for (int ii = i; ii < i + bm_dim && ii < m_dim; ii++) {
                    for (int jj = j; jj < j + bn_dim && jj < n_dim; jj++) {
                        for (int kk = k; kk < k + bk_dim && kk < k_dim; kk++) {
                            c_matrix[ii*n_dim + jj] += a_matrix[ii*k_dim+kk] * b_matrix[kk*n_dim+jj];
                        }
                    }
                }
            }
        }
    }
}
