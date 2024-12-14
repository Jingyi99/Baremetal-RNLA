#ifndef HOUSEHOLDER_QR_H
#define HOUSEHOLDER_QR_H

#include <stdio.h>
#include <stdlib.h>

// Function prototypes for test functions
void test_house();
void test_houseHolderHelper();
void test_houseHolderQR();
void test_backSubstitution();
void test_houseHolderQRLS();

// Function prototypes for the core functions used in the code
void* allocate_vector_clear(uint32_t bytes);
void vector_clear(void* arr, uint32_t bytes);
static inline float dot_product(float* x, float* y, int m);
float house(int m, float* x, float* v);
float* houseHolderHelper(float beta, float* v, int m);
void backSubstitution(float* R, float* y, float* x, int m, int n);
void houseHolderQRb(float* A, float* b, int m, int n);
float* householderQRLS(float* A, float* b, int m, int n);

#endif // HOUSEHOLDER_QR_H
