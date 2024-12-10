import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix, random

def print_array(name, data, data_size, data_type='double', data_fmt='{}', fold=10):
    print(f"{name} [{data_size}] = {{")
    for i in range(0, len(data), fold):
        print('  ', ', '.join(data_fmt.format(x) for x in data[i:i+fold]), ',', sep='')
    print('};')

# Read from MAT file
# info = sio.loadmat('b1_ss.mat')
# mat_A = info['Problem']['A']
# a_csc = mat_A[0][0]
# a_data = a_csc.data
# a_indices = a_csc.indices
# a_indptr = a_csc.indptr

# k_dim = a_csc.shape[0]
# n_dim = a_csc.shape[1]
# m_dim = n_dim
m_dim = 20
n_dim = 10
k_dim = 1

def print_header(dtype):
    print(f'''#define M_DIM {m_dim}
          #define K_DIM {k_dim}
          #define N_DIM {n_dim}
          
          typedef {dtype} data_t;
          ''')

def generate_full_rank_matrix(m_dim, n_dim):
    while True:
        A = np.random.rand(m_dim, n_dim)
        if np.linalg.matrix_rank(A) == min(m_dim, n_dim):
            return A
        

 
def generate_dense_ls(m_dim, n_dim):
    a_mat = generate_full_rank_matrix(m_dim, n_dim)
    b_vec = np.random.rand(m_dim)
    x, residuals, rank, s = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    print_header("double")
    print_array('static data_t a_matrix', a_mat.flatten(), 'M_DIM*N_DIM')
    print_array('static data_t b_vec', b_vec.flatten(), m_dim)
    print_array('static data_t x_vec', x.flatten(), n_dim)


def generate_sparse_dense(m_dim, n_dim, k_dim):
    a_mat = random(m_dim, k_dim, density=0.3, dtype=int, format="csc", data_rvs=lambda s: np.random.randint(-10, 11, size=s))
    a_indptr = a_mat.indptr
    a_data = a_mat.data
    a_indices = a_mat.indices
    b_matrix = np.random.randint(-10, 11, size=(k_dim, n_dim))

    verify_data= np.array(np.dot(a_mat.todense(), b_matrix))
    # verify_data = a_mat.dot(b_matrix)
    print_header("int")
    print_array('static int a_matrix_indptr', a_indptr, n_dim+1)
    print_array('static int a_matrix_indices', a_indices, a_mat.nnz)
    print_array('static data_t a_matrix_data', a_data, a_mat.nnz)
    print_array('static data_t b_matrix', b_matrix.flatten(), 'M_DIM*K_DIM')
    print_array('static data_t verify_data', verify_data.flatten(), 'M_DIM*N_DIM')

def generate_dense_sparse(m_dim, n_dim, k_dim):
    b_mat = random(m_dim, k_dim, density=0.3, dtype=int, format="csc", data_rvs=lambda s: np.random.randint(-10, 11, size=s))
    b_indptr = b_mat.indptr
    b_data = b_mat.data
    b_indices = b_mat.indices
    a_matrix = np.random.randint(-10, 11, size=(k_dim, n_dim))

    verify_data= np.array(np.dot(a_matrix, b_mat.todense()))
    # verify_data = a_mat.dot(b_matrix)
    print_header("double")
    print_array('static data_t a_matrix', a_matrix.flatten(), 'M_DIM*K_DIM')
    print_array('static int b_matrix_indptr', b_indptr, n_dim+1)
    print_array('static int b_matrix_indices', b_indices, b_mat.nnz)
    print_array('static data_t b_matrix_data', b_data, b_mat.nnz)
    print_array('static data_t verify_data', verify_data.flatten(), 'M_DIM*N_DIM')

# generate_dense_sparse(m_dim, n_dim, k_dim)
generate_dense_ls(m_dim, n_dim)



# s_matrix = np.random.randint(-1, 1, size=(m_dim, k_dim))
# verify_data = np.matmul(s_matrix, a_csc.todense())
# verify_data = np.array(verify_data).flatten()
# # a_hat_matrix = np.dot(s_matrix, a_csc)

# def print_array(name, data, data_size, data_type='float', data_fmt='{}', fold=10):
#     print(f"{name} [{data_size}] = {{")
#     for i in range(0, len(data), fold):
#         print('  ', ', '.join(data_fmt.format(x) for x in data[i:i+fold]), ',', sep='')
#     print('};')

# print_array('static data_t a_matrix', s_matrix.flatten(), 'M_DIM*K_DIM')
# print_array('static int b_matrix_indptr', a_indptr, n_dim+1)
# print_array('static int b_matrix_indices', a_indices, a_csc.nnz)
# print_array('static data_t b_matrix_data', a_data, a_csc.nnz)
# print_array('static data_t verify_data', verify_data, 'M_DIM*N_DIM')