import numpy as np
import scipy.io as sio
import scipy.sparse as sp
# from gendata import print_array, print_header

def print_array(name, data, data_size, data_type='float', data_fmt='{}', fold=10):
    print(f"{name} [{data_size}] = {{")
    for i in range(0, len(data), fold):
        print('  ', ', '.join(data_fmt.format(x) for x in data[i:i+fold]), ',', sep='')
    print('};')

def print_header(dtype, m_dim, k_dim, n_dim):
    print(f'''#define M_DIM {m_dim}
#define K_DIM {k_dim}
#define N_DIM {n_dim}

typedef {dtype} data_t;
''')

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
    return a_data, a_indices, a_indptr, m_dim, n_dim

file = 'mk12-b2.mat'
a_data, a_indices, a_indptr, m_dim, n_dim = read_mat(file)


