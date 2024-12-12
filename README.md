# Baremetal-RNLA
A library for randomized numerical linear algebra routines intended to run in baremetal runtime on RISC-V research chips

### Building for RISC-V

```bash
# make sure $RISCV is set
cmake -S ./ -B ./build/
cmake --build ./build
spike ./build/test_cpu
```


### Building for RISC-V Vector

```bash
# make sure $RISCV is set
cmake -S ./ -B ./build/ -D RISCV_V=ON
cmake --build ./build
spike --isa=rv64gcv_zicntr --varch=vlen:512,elen:32 ./build/test_rvv
```

NOTE: For large matrices you need to increase memory available for spike AND increase heap size in linker script htif.ld. 
- To increase spike memory allocation use `-m` flag (run `spike -h` to see details)
- To increase heap size: change line 130 (`PROVIDE(__heap_size = 128K);  `) to appropriate size

### Dataset generation script

```bash 
# Pregenerate sketching matrix and input matrix
# type is the type of sketching matrix, either (-1, 1) 'interval 'or ±1 'fixed'
python dataset/gen_sketch_sparse.py <m_dim> <n_dim> <type> <filename> dataset/sketching_matrix/<testname>.h
```