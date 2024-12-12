import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix, random
import sys
from util import *
# python dataset/gensketch.py <d_dim> <m_dim> <type> > dataset/sketching_matrix/sk.h
# type: interval, fixed

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

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python gensketch.py <d_dim> <m_dim> <type>")
        sys.exit(1)
    
    d_dim = int(sys.argv[1])
    m_dim = int(sys.argv[2])
    type = sys.argv[3]
    
    generate_skecthing_matrix(d_dim, m_dim, type)
