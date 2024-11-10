#include <stdio.h>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Householder>


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

int main() {
  int r = 4;
  int c = 4;
  int i, j;

  Matrix4d Q;
  Vector4d b, sol;
  Matrix4d A = Matrix4d::Random();

  // Print Random Input Matrix
  printf("Random Input Matrix:\n");
  cout << A << endl;            

  // Perform QR Decomp with HouseHolder
  HouseholderQR<Matrix4d> qr(A);
  Q = qr.householderQ();
  cout << "Q of Householder QR Decomp" << endl;
  cout << Q << endl;

  // Solve simple equation
  b = Vector4d::Random();
  sol = A.householderQr().solve(b);
  cout << "Linear System Solve with Householder" << endl;
  cout << sol << endl;

  return 0;
}