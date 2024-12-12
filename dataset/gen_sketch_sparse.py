import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix, random
import sys
from util import *
from genSuitsparse import read_mat, generate_sparse_only
# python dataset/gen_sketch_sparse.py <d_dim> <m_dim> <type> <filename>> dataset/sketching_matrix/sk.h
# type: interval, fixed

m_dim = 30
n_dim = 20
type = 'interval'
filename = 'mk12-b2.mat'

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
    print_header2(d_dim, m_dim, "float")
    print_array('static data_t sketching_matrix', A.flatten(), 'd_DIM*m_DIM')
    return A

def generate_custom_sparse(m_dim, n_dim):
    a_mat = random(m_dim, n_dim, density=0.3, dtype=int, format="csc", data_rvs=lambda s: np.random.randint(-100, 100, size=s))
    a_indptr = a_mat.indptr
    a_data = a_mat.data
    a_indices = a_mat.indices
    b_vec = np.random.rand(m_dim)
    print_header2(m_dim, n_dim)
    print_array('static int a_matrix_indptr', a_indptr, n_dim+1)
    print_array('static int a_matrix_indices', a_indices, len(a_data))
    print_array('static data_t a_matrix_data', a_data, len(a_data))
    print_array('static data_t b_vec', b_vec.flatten(), m_dim)

if __name__ == '__main__':
    if len(sys.argv) == 4:
        n_dim = int(sys.argv[2])    
        d_dim = n_dim  * 3
        m_dim = int(sys.argv[1]) 
        type = sys.argv[3]
        filename = sys.argv[4]
    
        generate_skecthing_matrix(d_dim, m_dim, type)
        generate_sparse_only(filename)
        
    elif len(sys.argv) == 3:
        n_dim = int(sys.argv[2])    
        d_dim = n_dim  * 3
        m_dim = int(sys.argv[1]) 
        type = sys.argv[3]
    
        generate_skecthing_matrix(d_dim, m_dim, type)
        generate_custom_sparse()
