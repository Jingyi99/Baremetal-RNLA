cmake_minimum_required(VERSION 3.15.0)

project(bmrnla VERSION 0.1 LANGUAGES C)

include(${CMAKE_SOURCE_DIR}/rnla.cmake)

set(BMRNLA_CMAKE_PATH_INCLUDE_PATH ${BMRNLA_CMAKE_PATH}/src/${USE_TYPE} CACHE STRING "BMRNLA include path")

option(RISCV_V              "Use RISCV vector extension"        ON)
option(NANO                 "Use htif_nano only (supports printf with floats)" OFF)

add_library(target-riscv INTERFACE)
target_compile_features(target-riscv INTERFACE c_std_11)
target_compile_definitions(target-riscv INTERFACE RISCV)

set(WRAP_SPECS_FILE     "htif_wrap.specs")
set(SPECS_FILE          "htif_nano.specs")
set(SPEC_FLAGS          -specs=${SPECS_FILE} -specs=${WRAP_SPECS_FILE})

if(NANO) # Override default
  set(SPEC_FLAGS        -specs=${SPECS_FILE})
endif()

set(ARCH               "rv64gc")
set(ABI                "lp64d")
set(CMODEL             "medany")

if (RISCV_V)
  list(APPEND ARCH "v")
endif()

list(JOIN ARCH "" ARCH)

if (NOT DEFINED LINKER_SCRIPT)
  set(LINKER_SCRIPT     ${CMAKE_SOURCE_DIR}/src/rvv/htif.ld)
endif()

target_compile_options(target-riscv INTERFACE -O1 -fno-common -fno-builtin-printf)
target_compile_options(target-riscv INTERFACE -static -u _printf_float)
target_compile_options(target-riscv INTERFACE -march=${ARCH} -mabi=${ABI} -mcmodel=${CMODEL})
target_compile_options(target-riscv INTERFACE -Wl,-Map=output.map ${SPEC_FLAGS})
target_compile_options(target-riscv INTERFACE -T ${LINKER_SCRIPT})
target_compile_options(target-riscv INTERFACE -ffast-math)

target_link_options(target-riscv INTERFACE -static -u _printf_float)
target_link_options(target-riscv INTERFACE -fno-common -fno-builtin-printf)
target_link_options(target-riscv INTERFACE -march=${ARCH} -mabi=${ABI} -mcmodel=${CMODEL})
target_link_options(target-riscv INTERFACE -Wl,-Map=output.map ${SPEC_FLAGS})
target_link_options(target-riscv INTERFACE -T ${LINKER_SCRIPT})
target_link_options(target-riscv INTERFACE -ffast-math)


message(STATUS "${CMAKE_SOURCE_DIR}")

# add_executable(test_cpu test/cpu/test_cpu.c)
add_executable(test_cpu src/cpu/sketch_n_solve.c)
add_executable(test_rvv test/rvv/test_rvv.c)

if (RISCV)
  message(STATUS "Building for RISC-V")
  set(cpu_impl
  ./src/cpu/gemm.c
  ./src/cpu/dsgemm.c
  ./src/cpu/randgen.c
  ./src/cpu/ls_qr_householder.c
  )
endif()

if (RISCV_V)
  message(STATUS "Using RVV implementation")
  add_compile_definitions(RISCV_V)
  set(rvv_impl
    ./src/rvv/gemm.c
    ./src/rvv/dsgemm.c
    ./src/rvv/rand/xorshift.c
    ./src/rvv/rand/xoroshiro128p.c
    ./src/rvv/ls_qr_householder.c
    ./src/rvv/sketch_n_solve.c
  )
endif()


add_library(bmrnla
    ${rvv_impl}
    ${cpu_impl}
)

target_link_libraries(bmrnla target-riscv)
target_include_directories(bmrnla PUBLIC
  ./test/
  ./dataset
  ./dataset/suitsparse
  ./src/rvv/headers
)

target_link_libraries(test_rvv PUBLIC bmrnla)
target_include_directories(test_rvv PUBLIC
  ./test/
  ./dataset
)

target_link_libraries(test_cpu PUBLIC bmrnla)
target_include_directories(test_cpu PUBLIC
  ./test/
  ./dataset
)

