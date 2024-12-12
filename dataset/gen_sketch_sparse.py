import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix, random
import sys
from util import *
from genSuitsparse import read_mat, generate_sparse_only
# python dataset/gen_sketch_sparse.py <d_dim> <m_dim> <type> <filename>> dataset/sketching_matrix/sk.h
# type: interval, fixed


density = 0.1

def generate_full_rank_matrix(m_dim, n_dim):
    while True:
        A = np.random.rand(m_dim, n_dim)
        if np.linalg.matrix_rank(A) == min(m_dim, n_dim):
            return A
        
def generate_skecthing_matrix(d_dim, m_dim, type):
    if type == 'interval':
        A = 2 * np.random.rand(d_dim, m_dim) - 1
    elif type == 'fixed':
        A =  np.random.choice([1, -1], size=(d_dim, m_dim))
  
    return A

def generate_custom_sparse(m_dim, n_dim, d_dim, type):
    sketching_matrix = generate_skecthing_matrix(d_dim, m_dim, type)
    print_header3(m_dim, n_dim, d_dim, 'float')
    print_array('static data_t sketching_matrix', sketching_matrix.flatten(), 'D_DIM*M_DIM')
    a_mat = random(m_dim, n_dim, density=0.1, dtype=int, format="csc", data_rvs=lambda s: np.random.randint(-100, 100, size=s))
    a_indptr = a_mat.indptr
    a_data = a_mat.data
    a_indices = a_mat.indices
    b_vec = np.random.rand(m_dim)
    print_array('static int a_matrix_indptr', a_indptr, n_dim + 1)
    print_array('static int a_matrix_indices', a_indices, len(a_data))
    print_array('static data_t a_matrix_data', a_data, len(a_data))
    print_array('static data_t b_vec', b_vec.flatten(), m_dim)

def generate_defined_matrix(a_data, a_indices, a_indptr, m_dim, n_dim, d_dim, nnz):
    b_vec = np.random.rand(m_dim)
    sketching_matrix = generate_skecthing_matrix(d_dim, m_dim, type)
    print_header3(m_dim, n_dim, d_dim, 'float')
    print_array('static data_t sketching_matrix', sketching_matrix.flatten(), 'D_DIM*M_DIM')
    print_array('static int a_matrix_indptr', a_indptr, n_dim+1)
    print_array('static int a_matrix_indices', a_indices, nnz)
    print_array('static data_t a_matrix_data', a_data, nnz)
    print_array('static data_t b_vec', b_vec.flatten(), m_dim)


if __name__ == '__main__':
    if len(sys.argv) == 5:
        type = sys.argv[3]
        filename = sys.argv[4]
        a_data, a_indices, a_indptr, m_dim, n_dim, nnz = read_mat(filename)
        d_dim = n_dim * 3
        
        generate_defined_matrix(a_data, a_indices, a_indptr, m_dim, n_dim, d_dim, nnz)
        
    elif len(sys.argv) == 4:
        n_dim = int(sys.argv[2])    
        d_dim = n_dim  * 3
        m_dim = int(sys.argv[1]) 
        type = sys.argv[3]
        
        generate_custom_sparse(m_dim, n_dim, d_dim, type)
