#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Function to compute the Euclidean norm of a vector */
float norm(int n, float* v) {
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

/* Function to compute the QR decomposition using Householder transformations */
void qr_householder(int m, int n, float** A, float** Q, float** R) {
    // Initialize R as a copy of A and Q as the identity matrix
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            R[i][j] = A[i][j];
        }
        for (int j = 0; j < m; j++) {
            Q[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int k = 0; k < n && k < m - 1; k++) {
        // Form the Householder vector
        float alpha = (R[k][k] >= 0) ? -norm(m - k, &R[k][k]) : norm(m - k, &R[k][k]);
        float r = sqrt(2.0 * alpha * (alpha - R[k][k]));
        float* v = (float*)calloc(m - k, sizeof(float));
        v[0] = R[k][k] - alpha;
        for (int i = 1; i < m - k; i++) {
            v[i] = R[k + i][k];
        }
        for (int i = 0; i < m - k; i++) {
            v[i] /= r;
        }

        // Apply the transformation to R
        for (int j = k; j < n; j++) {
            float dot = 0.0;
            for (int i = 0; i < m - k; i++) {
                dot += v[i] * R[k + i][j];
            }
            for (int i = 0; i < m - k; i++) {
                R[k + i][j] -= 2.0 * dot * v[i];
            }
        }

        // Update Q
        for (int i = 0; i < m; i++) {
            float dot = 0.0;
            for (int j = 0; j < m - k; j++) {
                dot += v[j] * Q[i][k + j];
            }
            for (int j = 0; j < m - k; j++) {
                Q[i][k + j] -= 2.0 * dot * v[j];
            }
        }

        free(v);
    }

    // Transpose Q
    for (int i = 0; i < m; i++) {
        for (int j = i + 1; j < m; j++) {
            float temp = Q[i][j];
            Q[i][j] = Q[j][i];
            Q[j][i] = temp;
        }
    }
}

/* Function to solve Rx = Q^T * b for x */
void solve(int n, float** R, float* Qt_b, float* x) {
    for (int i = n - 1; i >= 0; i--) {
        x[i] = Qt_b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= R[i][j] * x[j];
        }
        x[i] /= R[i][i];
    }
}

/* Least Squares Solver */
void least_squares(int m, int n, float** A, float* b, float* x) {
    float** Q = (float**)malloc(m * sizeof(float*));
    float** R = (float**)malloc(m * sizeof(float*));
    for (int i = 0; i < m; i++) {
        Q[i] = (float*)calloc(m, sizeof(float));
        R[i] = (float*)calloc(n, sizeof(float));
    }

    // QR decomposition
    qr_householder(m, n, A, Q, R);

    // Compute Q^T * b
    float* Qt_b = (float*)calloc(m, sizeof(float));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            Qt_b[i] += Q[j][i] * b[j];
        }
    }

    // Solve for x
    solve(n, R, Qt_b, x);

    // Free memory
    free(Qt_b);
    for (int i = 0; i < m; i++) {
        free(Q[i]);
        free(R[i]);
    }
    free(Q);
    free(R);
}

/* Utility function to create a 2D array */
float** create_matrix(int rows, int cols) {
    float** mat = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (float*)calloc(cols, sizeof(float));
    }
    return mat;
}

/* Main function */
int main() {
    int m = 4, n = 2;
    float** A = create_matrix(m, n);
    float b[] = {1, 2, 3, 4};
    float x[n];

    // Example matrix
    A[0][0] = 1; A[0][1] = 2;
    A[1][0] = 3; A[1][1] = 4;
    A[2][0] = 5; A[2][1] = 6;
    A[3][0] = 7; A[3][1] = 8;

    // Solve least squares
    least_squares(m, n, A, b, x);

    // Output result
    printf("Solution:\n");
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %f\n", i, x[i]);
    }

    // Free memory
    for (int i = 0; i < m; i++) {
        free(A[i]);
    }
    free(A);

    return 0;
}
