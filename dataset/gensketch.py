import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix, random
import sys

# python dataset/gensketch.py <d_dim> <m_dim> <type> > dataset/sketching_matrix/sk.h
# type: random, one, minus_one

def print_array(name, data, data_size, data_type='double', data_fmt='{}', fold=10):
    print(f"{name} [{data_size}] = {{")
    for i in range(0, len(data), fold):
        print('  ', ', '.join(data_fmt.format(x) for x in data[i:i+fold]), ',', sep='')
    print('};')

def print_header(d_dim, m_dim, dtype):
    print(f'''#define M_DIM {m_dim}
#define d_DIM {d_dim}
#define m_DIM {m_dim}

typedef {dtype} data_t;
''')

def generate_full_rank_matrix(m_dim, n_dim):
    while True:
        A = np.random.rand(m_dim, n_dim)
        if np.linalg.matrix_rank(A) == min(m_dim, n_dim):
            return A
        
def generate_skecthing_matrix(d_dim, m_dim, type):
    if type == 'random':
        A = 2 * np.random.rand(d_dim, m_dim) - 1
    elif type == 'one':
        A = np.ones((d_dim, m_dim)) 
    elif type == 'minus_one':
        A = np.ones((d_dim, m_dim)) * -1
    print_header(d_dim, m_dim, "double")
    print_array('static data_t sketching_matrix', A.flatten(), 'D_DIM*M_DIM')
    return A

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python gensketch.py <d_dim> <m_dim> <type>")
        sys.exit(1)
    
    d_dim = int(sys.argv[1])
    m_dim = int(sys.argv[2])
    type = sys.argv[3]
    
    generate_skecthing_matrix(d_dim, m_dim, type)
