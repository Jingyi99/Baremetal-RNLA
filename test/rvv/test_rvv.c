#include "lib.h"
#include <stdio.h>
#include <math.h>
#include <riscv.h>
#include "gemm_rvv.h"
#include "householder.h"

int main() {
    // test_dsgemm_rvv();
    // test_house();
    // test_houseHolderHelper();
    // test_houseHolderQR();
    // test_backSubstitution();
    test_houseHolderQRLS();
}