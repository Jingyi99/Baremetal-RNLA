// dense(left matrix) sparse(right matrix) matmul in CSC format
void dsgemm_csc(float *c_matrix, const float *a_matrix,
            const int *b_matrix_indptr,
            const int *b_matrix_indices, const float *b_matrix_data,
            const unsigned int m_dim, const unsigned int n_dim,
            const unsigned int k_dim);

// sparse(left matrix) dense(right matrix) matmul in CSC format
void sdgemm_csc(int *c_matrix, const int *a_matrix_indptr,
            const int *a_matrix_indices, const int *a_matrix_data,
            const int *b_matrix,
            const unsigned int m_dim, const unsigned int n_dim,
            const unsigned int k_dim);

