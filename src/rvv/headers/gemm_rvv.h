#pragma once
#ifndef GEMM_RVV_H_
#define GEMM_RVV_H_

float* householderQRLS(float* A, float* b, int m, int n);

void gemm_rvv(float *c_matrix, const float *a_matrix, const float *b_matrix, 
              const unsigned int m_dim, const unsigned int n_dim, const unsigned int k_dim);

void dsgemm_csc_rvv(float *c_matrix, const float *a_matrix,  const int *b_matrix_indptr, const int *b_matrix_indices, 
                    const float *b_matrix_data, const unsigned int m_dim, const unsigned int n_dim, const unsigned int k_dim);

#endif 