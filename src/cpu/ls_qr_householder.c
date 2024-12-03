#include <stdio.h>
#include <math.h>
#include <string.h>

float house(int m, float* x, float* v) {
    float sigma;
    float beta;
    for (int i = 1; i < m; i++) {
        sigma += x[i] * x[i];
    }
    memcpy(v, x, m * sizeof(float));
    v[0] = 1;
    if (sigma == 0 && x[0] >= 0) {
        beta = 0;
    } else if (sigma == 0 && x[0] < 0) {
        beta = -2;
    } else {
        float mu = sqrt(x[0]*x[0] + sigma);
        if (x[0] <= 0) {
            v[0] = x[0] - mu;
        } else {
            v[0] = -sigma / (x[0] + mu);
        }
        beta = 2 * v[0] * v[0] / (sigma + v[0]*v[0]);
        float v0 = v[0];
        for (int i = 0; i < m; i++) {
            v[i] = v[i]/v0;
        }
    }
    return beta;
}

void test_house() {
    float x[3] = {1, 2, 2};
    float v[3] = {0, 0, 0};
    float beta = house(3, x, v);
    for (int i = 0; i < 3; i++) {
        printf("x: %f, v: %f\n", x[i], v[i]);
    }
    printf("beta: %f\n", beta);
}

int main() {
    test_house();
}
// void least_squares(int m, int n, float * a_matrix, float* b, float *x ) {
//     for (int j = 0; j < n; j++) {

//     }
// }