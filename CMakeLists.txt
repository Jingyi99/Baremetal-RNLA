cmake_minimum_required(VERSION 3.15.0)

project(bmrnla VERSION 0.1 LANGUAGES C)

include(${CMAKE_SOURCE_DIR}/rnla.cmake)

set(BMRNLA_CMAKE_PATH_INCLUDE_PATH ${BMRNLA_CMAKE_PATH}/src/${USE_TYPE} CACHE STRING "BMRNLA include path")

option(RISCV_V              "Use RISCV vector extension"        ON)
option(NANO                 "Use htif_nano only (supports printf with floats)" ON)

add_library(target-riscv INTERFACE)
target_compile_features(target-riscv INTERFACE c_std_11)
target_compile_definitions(target-riscv INTERFACE RISCV)

set(WRAP_SPECS_FILE     "htif_wrap.specs")
set(SPECS_FILE          "htif_nano.specs")
set(SPEC_FLAGS          -specs=${SPECS_FILE} -specs=${WRAP_SPECS_FILE})
set(ARCH                "rv64gc")
set(ABI                 "lp64d")
set(CMODEL              "medany")

if(NANO) # Override default (allows printf in spike)
  set(SPEC_FLAGS        -specs=${SPECS_FILE})
endif()

if (RISCV_V)
  list(APPEND ARCH "v")
endif()

if (NOT DEFINED LINKER_SCRIPT)
  set(LINKER_SCRIPT     ${CMAKE_SOURCE_DIR}/src/rvv/htif.ld)
endif()

list(JOIN ARCH "" ARCH)


target_compile_options(target-riscv INTERFACE -O1 -g -fno-common -fno-builtin-printf)
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
    ./src/cpu/gemm.c
    ./src/rvv/c/gemm.c
    ./src/rvv/c/dsgemm.c
    ./src/rvv/c/rand/xorshift_m8.c
    ./src/rvv/c/rand/xoroshiro128p.c
    ./src/rvv/c/ls_qr_householder.c
  )
endif()

# Add targets for different executables
#add_executable(test_cpu src/cpu/sketch_n_solve.c)
add_executable(test_cpu src/cpu/pregen_sketch_solve.c ${cpu_impl})
#add_executable(test_cpu src/cpu/time_rand.c ${cpu_impl})
add_executable(test_rvv test/rvv/test_rvv.c test/rvv/ls_qr_householder_test.c)
add_executable(predefined_sns_rvv ${rvv_impl} ./src/rvv/c/predefined_sns_rvv.c)
add_executable(sns_rvv ${rvv_impl} ./src/rvv/c/sns_rvv.c)

# Add library for both cpu and rvv sources
add_library(bmrnla ${rvv_impl}
${cpu_impl})

# Create header variable for all RVV targall RVV
set(rvv_header_dirs
  ./test/
  ./dataset
  ./dataset/ls
  ./dataset/test_mats
  ./dataset/suitsparse
  ./dataset/sketching_matrix
  ./src/rvv/headers
)

# Set link libraries and include directories for targets
target_link_libraries(bmrnla target-riscv)
target_include_directories(bmrnla PUBLIC ${rvv_header_dirs})

target_link_libraries(predefined_sns_rvv PUBLIC bmrnla)
target_include_directories(predefined_sns_rvv PUBLIC ${rvv_header_dirs})

target_link_libraries(sns_rvv target-riscv)
target_include_directories(sns_rvv PUBLIC ${rvv_header_dirs})

target_link_libraries(test_rvv PUBLIC bmrnla)
target_include_directories(test_rvv PUBLIC ${rvv_header_dirs})

#target_link_libraries(test_cpu PUBLIC bmrnla)
target_link_libraries(test_cpu PUBLIC target-riscv)
target_include_directories(test_cpu PUBLIC
  ./test/
  ./dataset
  ./dataset/ls
  ./dataset/test_mats
)

set(OBJDUMP_FLAGS  -C -D -S --disassemble-all --disassemble-zeroes --section=.text --section=.text.startup --section=.text.init --section=.data)
add_custom_command(TARGET sns_rvv POST_BUILD COMMAND ${CMAKE_OBJDUMP} ${OBJDUMP_FLAGS}  sns_rvv > sns_rvv.dump)
add_custom_command(TARGET predefined_sns_rvv POST_BUILD COMMAND ${CMAKE_OBJDUMP} ${OBJDUMP_FLAGS}  predefined_sns_rvv > predefined_sns_rvv.dump)
add_custom_command(TARGET test_rvv POST_BUILD COMMAND ${CMAKE_OBJDUMP} ${OBJDUMP_FLAGS}  test_rvv > test_rvv.dump)
add_custom_command(TARGET test_cpu POST_BUILD COMMAND ${CMAKE_OBJDUMP} ${OBJDUMP_FLAGS}  test_cpu > test_cpu.dump)

