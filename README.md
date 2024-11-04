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
cmake -S ./ -B ./build/
cmake --build ./build
spike --isa=rv64gcv_zicntr --varch=vlen:512,elen:32 ./build/test_rvv
```
