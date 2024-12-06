#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>


// C style function to print a matrix
void print_matrix(uint32_t* A, int r, int c) {
  uint32_t i, j;
  for(i = 0; i < r; i++) {
    for(j = 0; j < c-1; j++) {
      printf("%d ", *(A + i*c + j));
    }
    printf("%d\n", *(A + i*c + j));
  }
}

void print_csc_matrix(uint32_t nnz, uint32_t N, uint32_t* csc_mat) {
  int x;

  uint32_t* row_ind = csc_mat;
  uint32_t* vals = csc_mat + nnz;
  uint32_t* col_ptr = csc_mat + 2*nnz;

  printf("row_ind: ");
  for(x=0; x < nnz; x++) {
    printf("%d ", row_ind[x]);
  }
  printf("\n");

  printf("vals: ");
  for( x=0; x < nnz; x++) {
    printf("%d ", vals[x]);
  }
  printf("\n");

  printf("col_ptr: ");
  for( x=0; x < N+1; x++) {
    printf("%d ", col_ptr[x]);
  }
  printf("\n");

}

 // Convert CSC to dense matrix
void csc2dense(uint32_t* dense_matrix, uint32_t* csc_mat, uint32_t nnz, uint32_t M, uint32_t N) {
  // Zero the result array
  for(uint8_t i = 0; i < M; i++) {
    for(uint8_t j = 0; j < N; j++) {
      dense_matrix[i*N + j] = 0;
    }
  }

  uint32_t* row_indices = csc_mat;
  uint32_t* values      = csc_mat+nnz;
  uint32_t* col_ptr     = csc_mat+2*nnz;

  print_csc_matrix(nnz, N, csc_mat);

  for (uint32_t col = 0; col < N; col++) {
    for (uint32_t row_ind = col_ptr[col]; row_ind < col_ptr[col + 1]; row_ind++) {
      uint32_t row = row_indices[row_ind];
      dense_matrix[row*N + col] = values[row_ind];
      printf("row=%d, col=%d, values=%d\n", row, col, values[row_ind]);
    }
  }

  print_matrix(dense_matrix, N, N);
  printf("\n");
}


/*
 * Generate an eye matrix is csc
 *
 * n - dimenion of matrix
 */
void eye_csc(uint32_t* eye_sparse, uint32_t N) {
  // Store as eye_sparse = [ eye_row | eye_data | eye_col ]

  // Zero array
  for(uint8_t x=0; x < 3*N+1; x++) {
    eye_sparse[x] = 0;
  }

  uint32_t nnz = N;
  uint32_t * eye_row   = eye_sparse;
  uint32_t * eye_data  = eye_sparse + nnz;
  uint32_t * eye_col   = eye_sparse + 2*nnz;

  for(uint8_t x=0; x < nnz; x++) {
    eye_data[x] = 1;
    eye_col[x] = x;
    eye_row[x] = x;
  }
  eye_col[N] = nnz;
}

/*
 * Generate a shift matrix
 */
void shift_csc(uint32_t* mat_sparse, uint32_t N, bool left) {

  // Store as mat_sparse = [mat_row | mat_col | mat_data]
  for(uint8_t x=0; x < 3*N-1; x++) { // Zero array
    mat_sparse[x] = 0;
  }
    
  uint32_t nnz = N-1;
  uint32_t* mat_row  = mat_sparse;
  uint32_t* mat_data = mat_sparse + nnz;
  uint32_t* mat_col  = mat_sparse + 2*nnz;

  uint8_t x;
  uint8_t col_ind;
  for(x = 0; x < nnz; x++) {
    if (left) {
      mat_col[x] = x;
      mat_row[x] = x+1;
      col_ind = x;
    } else {
      mat_col[x+1] = x;
      mat_row[x]   = x;
      col_ind = x + 1;
    }

    mat_data[x] = 1;
  }

  // Fill rest of array with NNZ
  for(x=col_ind+1; x < N+1; x++){
    mat_col[x] = nnz;
  }

  // Last element is always NNZ
  // mat_col[N] = nnz;
}


void 

/*
 * For testing only
 */
int main() {

  uint32_t N = 8;
  uint32_t nnz = N-1;
  uint32_t N_tot = 3*N - 1; // 2*nnz + N + 1

  uint32_t eye[3*N+1];
  uint32_t shift_left[3*N-1];
  uint32_t shift_right[3*N-1];

  uint32_t dense_eye[N*N];
  uint32_t dense_shiftL[N*N];
  uint32_t dense_shiftR[N*N];


  eye_csc(eye, N);
  shift_csc(shift_left,  N, true);
  shift_csc(shift_right, N, false);

  printf("ashdufhl;adf\n");

  csc2dense(dense_eye, eye, N, N, N);
  csc2dense(dense_shiftL, shift_left, nnz, N, N);
  csc2dense(dense_shiftR, shift_right, nnz, N, N);
  // printf("ashdufhl;adf\n");




  // print_matrix(dense_shiftL, nnz, nnz);
}


//  // Convert CSC to dense matrix
// uint32_t* csc_print(int M, int N, uint32_t* col_pointers, uint32_t* row_indices, uint32_t values) {
//   // uint32_t dense_matrix[M][N] = {0};

//   int num_cols = n;

//   for (int col = 0; col < M; col++) {
//     for (int row = 0; row < N; row++) {
//       if (row < col_pointers[col] || row > col_pointers[col]) {

//       } else {
//         print();
//       }
//       int row = row_indices[i];
//       dense_matrix[row][col] = values[i];
//     }
//   }
// }
