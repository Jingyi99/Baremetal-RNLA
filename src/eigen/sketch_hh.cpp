#include <stdio.h>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Householder>
#include "dsgemm.h"
#include "small_sqL.h"

using namespace std;
using namespace Eigen;

// C style function to print a matrix
void print_matrix(MatrixXd A, int r, int c) {
  int i, j;
  for(i = 0; i < r; i++) {
    for(j = 0; j < c-1; j++) {
      printf("%.4f, ", A(i,j));
    }
    printf("%.4f\n", A(i,j));
  }
}

int main(int argc, char* argv[1]) {


  // Sketched HH QR Decompoistion Solver
  // Create empty sketch matrix
  float SA_[M][N] = {0};

  // Convert 2D arrays to Eigen matrices
  Matrix<float,P,M> S = Eigen::Map<Eigen::Matrix<float,P,M,RowMajor>> (S_dense[0]);
  Matrix<float,M,K> EA = Eigen::Map<Eigen::Matrix<float,M,K,RowMajor>> (A[0]);

  // Sketch input matrix
  dsgemm_csc(SA_[0], A[0], S_cptr, S_ind, S_data, M, N, K);

  // Solve linear system (least square Householder)
  Matrix<float,P,K> H  = Eigen::Map<Eigen::Matrix<float,P,K,RowMajor>> (SA_[0]);
  Matrix<float,P,N> SB_ = Eigen::Map<Eigen::Matrix<float,P,N,RowMajor>> (SB[0]);

  // Bypass user-implemented sparsematrix-matrix multiply
  //H = S*EA;

  cout << "Sketched A (SA)" << endl;
  cout << H << "\n" << endl;

  cout << "Sketched B (SB)" << endl;
  cout << SB_ << "\n" << endl;

  Matrix<float,N,P> X_sol1 = H.householderQr().solve(SB_);
  cout << "Linear Sketched System Solve with Householder" << endl;
  cout << X_sol1 << endl;


  // Nonsketched QR HH Decompositon Solver
  Matrix<float,M,N> EB = Eigen::Map<Eigen::Matrix<float,M,N,RowMajor>> (B[0]);
  Matrix<float,N,P> X_sol0 = EA.householderQr().solve(EB);

  // Verification
  Matrix<float,N,P> diff = X_sol0 - X_sol1;
  float tot_err = diff.cwiseAbs().sum();

  cout << "Linear System Solve with Householder" << endl;
  cout << X_sol0 << "\n" << endl;


  cout << "Diff matrix:_" << endl;
  cout << diff << "\n"  << endl;
  cout << "Total error: " << tot_err << endl;

  return 0;
}
