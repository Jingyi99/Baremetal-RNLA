
def print_array(name, data, data_size, data_type='float', data_fmt='{}', fold=10):
    print(f"{name} [{data_size}] = {{")
    for i in range(0, len(data), fold):
        print('  ', ', '.join(data_fmt.format(x) for x in data[i:i+fold]), ',', sep='')
    print('};')

def print_header2(m_dim, n_dim, dtype='float'):
    print(f'''
#define M_DIM {m_dim}
#define N_DIM {n_dim}

typedef {dtype} data_t;
''')
    
def print_header3(m_dim, n_dim, k_dim, dtype):
    print(f'''
#define M_DIM {m_dim}
#define N_DIM {n_dim}
#define K_DIM {k_dim}

typedef {dtype} data_t;
''')