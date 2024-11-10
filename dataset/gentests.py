import sys
from textwrap import dedent
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix, random

def print_array(name, data, data_size, data_type='float', data_fmt='{:.6f}', fold=10, f=sys.stdout):
    data_str = ""
    for i in range(0, len(data), fold):
        data_str +=  ', '.join(data_fmt.format(x) for x in data[i:i+fold]) + ",\n"
    print(f"{name} [{data_size}] = {{ \n{data_str[:-2]} }};\n", file=f)

def print_mat_as_c_array(name, matrix, dim0, dim1, dtype='float', f=sys.stdout):
    print(f"{name} [{dim0}][{dim1}] = {{", file=f )
    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        e = ",\n" if i != matrix.shape[0]-1 else ""
        print("{", ", ".join(map(str, row)), "}", end=e, file=f)
    print("};\n", file=f)

"""
  Generates header for different testing inputs.
  S is PxM sketching matrix as sparse in CSC, A is dense
  MxK input matrix, SA is S applied to A a PxK matrix (for verification)
  B is the solution for the linear system, X is the solution a KxN matria
  (for verification)

    left - true, if sketching matrix is applied to left, applied to right others
    dtype -
    P, M, K, N - the appropos dimensions
"""
def generate_sparse_dense(left, P, M, K, N, dtype=float, f=sys.stdout):

    dtype_str = "float" if dtype==float else "int"
    gen_rand = lambda a, b : np.random.rand(a,b) if (dtype == float) else np.random.randint(-10, 11, size=(a, b))
    S = random(P, M, density=0.3, dtype=dtype, format="csc", data_rvs=lambda s: gen_rand(1,s).flatten())

    solved = False
    while(not solved):
        # Form known matrices
        A = gen_rand(M,K)
        B = gen_rand(M,N)
        SA = np.asarray(np.matmul(S.todense(),A))
        SB = np.asarray(np.matmul(S.todense(),B))

        #A = np.random.randint(-10, 11, size=(K, N))
        #SA = np.array(np.dot(S.todense(), A))
        #B = np.random.randint(-10, 11, size=(M,P))

        # Manual implementation of Househoulder linear solver
        Q,R = np.linalg.qr(SA)
        Y = np.matmul(Q.T, SB)
        solved = np.linalg.matrix_rank(R) == R.shape[0]

    print()
    print("A: ", A.shape)
    print("SA: ", SA.shape)
    print("B: ", B.shape)
    print("Q: ", Q.shape)
    print("R: ", R.shape)
    print("Y: ", Y.shape)

    #X = np.linalg.solve(R, Y)
    X, red, rank, Sing  = np.linalg.lstsq(R,Y)

    print("X: ", X.shape)

    # Write header file
    print(f'#define M {M}\n' +
          f'#define K {K}\n' +
          f'#define N {N}\n' +
          f'#define P {P}\n\n' +
          f'typedef {dtype_str} data_t;\n', file=f)
    print_array('static int S_cptr',    S.indptr,     M+1,   data_fmt='{}', f=f)
    print_array('static int S_ind',     S.indices,    S.nnz, data_fmt='{}', f=f)
    print_array('static data_t S_data', S.data,       S.nnz, data_fmt='{}', f=f)
    print_mat_as_c_array('static data_t A',  A,  'M', 'K', f=f)
    print_mat_as_c_array('static data_t B',  B,  'M', 'N', f=f)
    print_mat_as_c_array('static data_t X',  X,  'K', 'N', f=f)
    print_mat_as_c_array('static data_t SA', SA, 'P', 'K', f=f)
    print_mat_as_c_array('static data_t SB', SB, 'P', 'N', f=f)
    print_mat_as_c_array('static data_t S_dense',  np.array(S.todense()), 'P', 'M', f=f)

    return np.allclose(np.matmul(SA,X), SB)

def generate_dataset():

    dimdict = lambda P,M,K,N: {"P": P, "M": M, "K": K, "N": N}
    sizes = {"tiny" : dimdict(8, 10, 8, 10),
             "small": dimdict(13, 22, 13, 10),
             "med"  : dimdict(30, 50, 48, 30),
             "large": dimdict(150, 380, 256, 30),
             "giant": dimdict(300, 1000, 990, 30)}

    left = True   # True for left sketch, False for right
    square = True # True makes A square KxK, False KxN
    lr_char = "L" if left else "R"
    sq_char = "sq" if square else "rect"
    folder  = "square" if square else "rect"

    for key, v in sizes.items():
        print(key, end='...')
        header = open(f"./{folder}/{key}_{sq_char}{lr_char}.h", "w+")
        N = v["K"] if square else v["N"]
        res = generate_sparse_dense(left, v["P"], v['M'], v["K"], N, f=header)
        print(f"{res}",end="")
        print("...done")


generate_dataset()

