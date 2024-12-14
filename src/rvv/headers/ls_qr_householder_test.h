#ifndef TEST_HOUSEHOLDER_H
#define TEST_HOUSEHOLDER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// Function prototypes for test functions
void test_house();
void test_houseHolderHelper();
void test_houseHolderQR();
void test_backSubstitution();
void test_houseHolderQRLS();

// Function prototypes for the algorithms used in the tests (assumed to be implemented elsewhere)
float house(int m, float* x, float* v);
float* houseHolderHelper(float beta, float* v, int m);
void houseHolderQR(float* A, int m, int n);
void backSubstitution(float* R, float* y, float* x, int m, int n);
float* householderQRLS(float* A, float* b, int m, int n);

#endif // TEST_HOUSEHOLDER_H