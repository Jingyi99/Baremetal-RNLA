    void test_house() {
        // set test cases here
        int m = 3;
        float x[] = {2, 2, 1};
        float v[] = {0, 0, 0};
        // 
        float beta = house(m, x, v);
        for (int i = 0; i < m; i++) {
        printf("x: %f, v: %f\n", x[i], v[i]);
        }
        printf("beta: %f\n", beta);
    }

    void test_houseHolderHelper() {
        // set test case here
        int m = 3;
        float x[] = {2, 2, 1};
        // float v[] = {5, 2, 1};
        // float beta = 2.0 / 30.0;
        float v[] = {1, -2, -1};
        float beta = 1.0 / 3.0;
        // 
        float* result = houseHolderHelper(beta, v, m);

        printf("H:\n");
        for (int i = 0; i < m; i++){
        for (int j = 0; j < m; j++){
            printf("%f ", result[i*m+j]);
        }
        printf("\n");
        }
        float *Hx = malloc(sizeof(float) * m);
        gemm(Hx, result, x, m, 1, m);
        printf("Hx: \n");
        for (int i = 0; i < m; i++){
        printf("%f ", Hx[i]);
        }
        printf("\n");
        free(Hx);
    }

    void test_houseHolderQR() {
        // set test case here
        // int m = 4;
        // int n = 3;
        // float A[] = {1.0, -1.0, 4.0, 1.0, 4.0, -2.0, 1.0, 4.0, 2.0, 1.0, -1.0, 0.0};
        // int m = 3;
        // int n = 3;
        // float A[] = {2.0, -2.0, 18.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0};
        int m = 4;
        int n = 3;
        float A[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 10.0, 11.0, 13.0};
        printf("A:\n");
        for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            printf("%f ", A[i*n+j]);
        }
        printf("\n");
        }
        houseHolderQR(A, m, n);
        printf("R:\n");
        for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            printf("%f ", A[i*n+j]);
        }
        printf("\n");
        }
    }

    void test_backSubstitution() {
        int m = 4;
        int n = 3;
        float A[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 10.0, 11.0, 13.0};
        float b[] = {14.0, 32.0, 50.0, 68.0};
        float R[] = {12.884099 , 14.591630 , 16.299161 , 0.0, 1.041315 , 2.082630 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        float y[] = {90.964842, 8.330522 , 0.000000 , 0.000000 };
        householderQRLS(A, b, m, n);
        printf("A back: \n");
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                printf("%f ", A[i*n+j]);
            }
            printf("\n");
        }
        printf("b back: \n");
        for (int i = 0; i < m; i++){
            printf("%f ", b[i]);
        }
        float *x = backSubstitution(A, b, m, n);
        printf("x: \n");
        for (int i = 0; i < n; i++) {
        printf("%f ", x[i]);
        }
        printf("\n");
    }

    void test_houseHolderQRLS() {
        int m = 4;
        int n = 3;
        float A[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 10.0, 11.0, 13.0};
        float b[] = {14.0, 32.0, 50.0, 68.0};

        float *x = (float*) malloc(sizeof(float) * n);
        memset(x, 0, sizeof(float) * n);
        x = householderQRLS(A, b, m, n);
        printf("x: \n");
        for (int i = 0; i < n; i++) {
        printf("%lf ", x[i]);
        }
        printf("\n");
    }

    void test_backwarderror(float* A, float* Q, float* R, int m, int n){
        // print R 
        FILE *file = fopen("R_matrix.txt", "w");
        if (file == NULL) {
            fprintf(stderr, "Error opening file for writing\n");
            return;
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                fprintf(file, "%f ", R[i * n + j]);
            }
            fprintf(file, "\n");
        }
        fclose(file);
        // print Q
        file = fopen("Q_matrix.txt", "w");
        if (file == NULL) {
            fprintf(stderr, "Error opening file for writing\n");
            return;
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                fprintf(file, "%f ", Q[i * m + j]);
            }
            fprintf(file, "\n");
        }
        fclose(file);
        // print (A - qr) / A
        float* qr = (float*)malloc(m*n*sizeof(float));
        gemm(qr, Q, R, m, n, n);
        float error = 0;
        float residual = 0;
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                residual += (A[i*n+j] - qr[i*n+j]) * (A[i*n+j] - qr[i*n+j]);
            }
        }
        residual = sqrt(residual);
        printf("residual: %f\n", residual);
        float normA = 0;
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                normA += A[i*n+j] * A[i*n+j];
            }
        }
        normA = sqrt(normA);
        printf("normA: %f\n", normA);
        error = residual / normA;
        printf("backward error: %f\n", error);
    }

    // int main() {
    //     // test_house();
    //     // test_houseHolderHelper();
    //     // test_houseHolderQR();
    //     // test_backSubstitution();
    //     test_houseHolderQRLS();
    // }