import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from util import *
import sys

# Read from MAT file
def read_mat(filename):
    info = sio.loadmat(filename)
    mat_A = info['Problem']['A']
    a_csc = mat_A[0][0]
    a_data = a_csc.data
    a_indices = a_csc.indices
    a_indptr = a_csc.indptr

    m_dim = a_csc.shape[0]
    n_dim = a_csc.shape[1]
    nnz = len(a_data)
    return a_data, a_indices, a_indptr, m_dim, n_dim, nnz

# assume a is csc format
def generate_sparse_only(filename):
    a_data, a_indices, a_indptr, m_dim, n_dim, nnz = read_mat(filename)
    b_vec = np.random.rand(m_dim)
    print_header2(m_dim, n_dim)
    print_array('static int a_matrix_indptr', a_indptr, n_dim+1)
    print_array('static int a_matrix_indices', a_indices, nnz)
    print_array('static data_t a_matrix_data', a_data, nnz)
    print_array('static data_t b_vec', b_vec.flatten(), m_dim)

# filename = 'dataset/suitsparse/mk12-b1.mat'

# if __name__ == '__main__':
#     filename = sys.argv[1]
#     generate_sparse_only(filename)
