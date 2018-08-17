// mlLinearAlgebra.cpp
// Linear algebra related to machine learning
//
// Written by Bradley Denby
// Other contributors: None
//
// To the extent possible under law, the author(s) have dedicated all copyright
// and related and neighboring rights to this software to the public domain
// worldwide. This software is distributed without any warranty.
//
// You should have received a copy of the CC0 Public Domain Dedication with this
// software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.

// Standard libraries
#include <cstddef>  // size_t
#include <iostream> // cout
#include <ostream>  // endl

// Eigen
#include <Eigen/Core>                     // Eigen::Matrix<type,n1,n2>
#include <Eigen/Eigenvalues>              // 
#include <unsupported/Eigen/CXX11/Tensor> // Eigen::TensorFixedSize< type, Eigen::Sizes<n1,n2,n3> >

int main(int argc, char** argv) {

  // Scalars (doubles in this case); prefixed with "s"
  double sA = 1.0;
  double sB = 1.5;
  double sC = 2.0;
  std::cout << "Scalars:"
            << std::endl
            << "  sA: " << sA
            << std::endl
            << "  sB: " << sB
            << std::endl
            << "  sC: " << sC
            << std::endl << std::endl;

  // 2D Vectors (doubles); prefixed with "v2"
  Eigen::Matrix<double,2,1> v2A = Eigen::Matrix<double,2,1>::Random();
  Eigen::Matrix<double,2,1> v2B = Eigen::Matrix<double,2,1>::Random();
  Eigen::Matrix<double,2,1> v2C = Eigen::Matrix<double,2,1>::Random();
  std::cout << "2D Vectors:"
            << std::endl
            << "  v2A: " << std::endl
            << v2A
            << std::endl
            << "  v2B: " << std::endl
            << v2B
            << std::endl
            << "  v2C: " << std::endl
            << v2C
            << std::endl << std::endl;

  // 3D Vectors (doubles); prefixed with "v3"
  Eigen::Matrix<double,3,1> v3A = Eigen::Matrix<double,3,1>::Random();
  Eigen::Matrix<double,3,1> v3B = Eigen::Matrix<double,3,1>::Random();
  Eigen::Matrix<double,3,1> v3C = Eigen::Matrix<double,3,1>::Random();
  std::cout << "3D Vectors:"
            << std::endl
            << "  v3A: " << std::endl
            << v3A
            << std::endl
            << "  v3B: " << std::endl
            << v3B
            << std::endl
            << "  v3C: " << std::endl
            << v3C
            << std::endl << std::endl;

  // 2x2 Matrices (doubles); prefixed with "m2x2"
  Eigen::Matrix<double,2,2> m2x2A = Eigen::Matrix<double,2,2>::Random();
  Eigen::Matrix<double,2,2> m2x2B = Eigen::Matrix<double,2,2>::Random();
  Eigen::Matrix<double,2,2> m2x2C = Eigen::Matrix<double,2,2>::Random();
  std::cout << "2x2 Matrices:"
            << std::endl
            << "  m2x2A: " << std::endl
            << m2x2A
            << std::endl
            << "  m2x2B: " << std::endl
            << m2x2B
            << std::endl
            << "  m2x2C: " << std::endl
            << m2x2C
            << std::endl << std::endl;

  // 2x3 Matrices (doubles); prefixed with "m2x3"
  Eigen::Matrix<double,2,3> m2x3A = Eigen::Matrix<double,2,3>::Random();
  Eigen::Matrix<double,2,3> m2x3B = Eigen::Matrix<double,2,3>::Random();
  Eigen::Matrix<double,2,3> m2x3C = Eigen::Matrix<double,2,3>::Random();
  std::cout << "2x3 Matrices:"
            << std::endl
            << "  m2x3A: " << std::endl
            << m2x3A
            << std::endl
            << "  m2x3B: " << std::endl
            << m2x3B
            << std::endl
            << "  m2x3C: " << std::endl
            << m2x3C
            << std::endl << std::endl;

  // 3x2 Matrices (doubles); prefixed with "m3x2"
  Eigen::Matrix<double,3,2> m3x2A = Eigen::Matrix<double,3,2>::Random();
  Eigen::Matrix<double,3,2> m3x2B = Eigen::Matrix<double,3,2>::Random();
  Eigen::Matrix<double,3,2> m3x2C = Eigen::Matrix<double,3,2>::Random();
  std::cout << "3x2 Matrices:"
            << std::endl
            << "  m3x2A: " << std::endl
            << m3x2A
            << std::endl
            << "  m3x2B: " << std::endl
            << m3x2B
            << std::endl
            << "  m3x2C: " << std::endl
            << m3x2C
            << std::endl << std::endl;

  // 3x3x3 Tensors (doubles); prefixed with "t3x3x3"
  Eigen::TensorFixedSize< double, Eigen::Sizes<3,3,3> > t3x3x3A;
  t3x3x3A = t3x3x3A.random();
  std::cout << "3x3x3 Tensors:"
            << std::endl
            << "  t3x3x3A: "
            << std::endl;
  for(size_t i=0; i<3; i++) {
    std::cout << "    2D slice " << i << ":"
              << std::endl;
    for(size_t j=0; j<3; j++) {
      for(size_t k=0; k<3; k++) {
        std::cout << t3x3x3A(i,j,k) << " ";
      }
      std::cout << std::endl;
    }
  }
  Eigen::TensorFixedSize< double, Eigen::Sizes<3,3,3> > t3x3x3B;
  t3x3x3B = t3x3x3B.random();
  std::cout << "  t3x3x3B: "
            << std::endl;
  for(size_t i=0; i<3; i++) {
    std::cout << "    2D slice " << i << ":"
              << std::endl;
    for(size_t j=0; j<3; j++) {
      for(size_t k=0; k<3; k++) {
        std::cout << t3x3x3B(i,j,k) << " ";
      }
      std::cout << std::endl;
    }
  }
  Eigen::TensorFixedSize< double, Eigen::Sizes<3,3,3> > t3x3x3C;
  t3x3x3C = t3x3x3C.random();
  std::cout << "  t3x3x3C: "
            << std::endl;
  for(size_t i=0; i<3; i++) {
    std::cout << "    2D slice " << i << ":"
              << std::endl;
    for(size_t j=0; j<3; j++) {
      for(size_t k=0; k<3; k++) {
        std::cout << t3x3x3C(i,j,k) << " ";
      }
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;

  // Transpose
  std::cout << "2D Vector Transposes:"
            << std::endl
            << "  v2A: " << std::endl
            << v2A.transpose()
            << std::endl
            << "  v2B: " << std::endl
            << v2B.transpose()
            << std::endl
            << "  v2C: " << std::endl
            << v2C.transpose()
            << std::endl << std::endl;

  std::cout << "3D Vector Transposes:"
            << std::endl
            << "  v3A: " << std::endl
            << v3A.transpose()
            << std::endl
            << "  v3B: " << std::endl
            << v3B.transpose()
            << std::endl
            << "  v3C: " << std::endl
            << v3C.transpose()
            << std::endl << std::endl;

  std::cout << "2x2 Matrix Transposes:"
            << std::endl
            << "  m2x2A: " << std::endl
            << m2x2A.transpose()
            << std::endl
            << "  m2x2B: " << std::endl
            << m2x2B.transpose()
            << std::endl
            << "  m2x2C: " << std::endl
            << m2x2C.transpose()
            << std::endl << std::endl;

  std::cout << "2x3 Matrix Transposes:"
            << std::endl
            << "  m2x3A: " << std::endl
            << m2x3A.transpose()
            << std::endl
            << "  m2x3B: " << std::endl
            << m2x3B.transpose()
            << std::endl
            << "  m2x3C: " << std::endl
            << m2x3C.transpose()
            << std::endl << std::endl;

  std::cout << "3x2 Matrix Transposes:"
            << std::endl
            << "  m3x2A: " << std::endl
            << m3x2A.transpose()
            << std::endl
            << "  m3x2B: " << std::endl
            << m3x2B.transpose()
            << std::endl
            << "  m3x2C: " << std::endl
            << m3x2C.transpose()
            << std::endl << std::endl;

  std::cout << "3x3x3 Tensor Transposes: TODO"
            << std::endl << std::endl;

  // Addition
  std::cout << "2D Vector Addition:"
            << std::endl
            << "  v2A + v2B: " << std::endl
            << v2A + v2B
            << std::endl << std::endl;

  std::cout << "3D Vector Addition:"
            << std::endl
            << "  v3A + v3B: " << std::endl
            << v3A + v3B
            << std::endl << std::endl;

  std::cout << "2x2 Matrix Addition:"
            << std::endl
            << "  m2x2A + m2x2B: " << std::endl
            << m2x2A + m2x2B
            << std::endl << std::endl;

  std::cout << "2x3 Matrix Addition:"
            << std::endl
            << "  m2x3A + m2x3B: " << std::endl
            << m2x3A + m2x3B
            << std::endl << std::endl;

  std::cout << "3x2 Matrix Addition:"
            << std::endl
            << "  m3x2A + m3x2B: " << std::endl
            << m3x2A + m3x2B
            << std::endl << std::endl;

  std::cout << "3x3x3 Tensor Addition:"
            << std::endl;
  Eigen::TensorFixedSize< double, Eigen::Sizes<3,3,3> > t3x3x3ApB = t3x3x3A + t3x3x3B;
  for(size_t i=0; i<3; i++) {
    std::cout << "    2D slice " << i << ":"
              << std::endl;
    for(size_t j=0; j<3; j++) {
      for(size_t k=0; k<3; k++) {
        std::cout << t3x3x3ApB(i,j,k) << " ";
      }
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;

  // Multiplication
  std::cout << "2D Vector Dot Product:"
            << std::endl
            << "  v2A.dot(v2B): " << std::endl
            << v2A.dot(v2B)
            << std::endl << std::endl;

  std::cout << "3D Vector Dot Product:"
            << std::endl
            << "  v3A.dot(v3B): " << std::endl
            << v3A.dot(v3B)
            << std::endl << std::endl;

  std::cout << "2x2 Matrix Multiplication:"
            << std::endl
            << "  m2x2A * m2x2B: " << std::endl
            << m2x2A * m2x2B
            << std::endl << std::endl;

  std::cout << "2x3 and 3x2 Matrix Multiplication:"
            << std::endl
            << "  m2x3A * m3x2A: " << std::endl
            << m2x3A * m3x2A
            << std::endl << std::endl;

  std::cout << "3x2 and 2x3 Matrix Multiplication:"
            << std::endl
            << "  m3x2B * m2x3B: " << std::endl
            << m3x2B * m2x3B
            << std::endl << std::endl;

  std::cout << "3x3x3 Tensor Multiplication:"
            << std::endl;
  Eigen::TensorFixedSize< double, Eigen::Sizes<3,3,3> > t3x3x3AmB = t3x3x3A * t3x3x3B;
  for(size_t i=0; i<3; i++) {
    std::cout << "    2D slice " << i << ":"
              << std::endl;
    for(size_t j=0; j<3; j++) {
      for(size_t k=0; k<3; k++) {
        std::cout << t3x3x3AmB(i,j,k) << " ";
      }
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;

  // Hadamard Product
  std::cout << "2D Vector Hadamard Product:"
            << std::endl
            << "  v2A.array()*v2B.array(): " << std::endl
            << v2A.array()*v2B.array()
            << std::endl << std::endl;

  std::cout << "3D Vector Hadamard Product:"
            << std::endl
            << "  v3A.array()*v3B.array(): " << std::endl
            << v3A.array()*v3B.array()
            << std::endl << std::endl;

  std::cout << "2x2 Matrix Hadamard Product:"
            << std::endl
            << "  m2x2A.array()*m2x2B.array(): " << std::endl
            << m2x2A.array()*m2x2B.array()
            << std::endl << std::endl;

  std::cout << "2x3 Matrix Hadamard Product:"
            << std::endl
            << "  m2x3A.array()*m2x3B.array(): " << std::endl
            << m2x3A.array()*m2x3B.array()
            << std::endl << std::endl;

  std::cout << "3x2 Matrix Hadamard Product:"
            << std::endl
            << "  m3x2A.array()*m3x2B.array(): " << std::endl
            << m3x2A.array()*m3x2B.array()
            << std::endl << std::endl;

  std::cout << "3x3x3 Tensor Hadamard Product: TODO"
            << std::endl << std::endl;

  // Norm
  std::cout << "2D Vector Norms:"
            << std::endl
            << "  v2A.lpNorm<1>(): " << v2A.lpNorm<1>()
            << std::endl
            << "  v2A.lpNorm<2>(): " << v2A.lpNorm<2>()
            << std::endl
            << "  v2A.lpNorm<Eigen::Infinity>(): " << v2A.lpNorm<Eigen::Infinity>()
            << std::endl << std::endl;

  std::cout << "3D Vector Norms:"
            << std::endl
            << "  v3A.lpNorm<1>(): " << v3A.lpNorm<1>()
            << std::endl
            << "  v3A.lpNorm<2>(): " << v3A.lpNorm<2>()
            << std::endl
            << "  v3A.lpNorm<Eigen::Infinity>(): " << v3A.lpNorm<Eigen::Infinity>()
            << std::endl << std::endl;

  std::cout << "2x2 Matrix Norms: TODO"
            << std::endl << std::endl;

  std::cout << "2x3 Matrix Norms: TODO"
            << std::endl << std::endl;

  std::cout << "3x2 Matrix Norms: TODO"
            << std::endl << std::endl;

  std::cout << "3x3x3 Tensor Norms: TODO"
            << std::endl << std::endl;

  // Eigen Decomposition
  Eigen::EigenSolver< Eigen::Matrix<double,2,2> > m2x2Aes(m2x2A);
  std::cout << "2x2 Matrix Eigen Decomposition:"
            << std::endl
            << "  m2x2A eigenvalues: " << std::endl
            << m2x2Aes.eigenvalues()
            << std::endl
            << "  m2x2A eigenvectors: " << std::endl
            << m2x2Aes.eigenvectors()
            << std::endl << std::endl;

  // Singular Value Decomposition
  Eigen::JacobiSVD< Eigen::Matrix<double,2,3> > m2x3Ajsvd(m2x3A, Eigen::ComputeFullU|Eigen::ComputeFullV);
  std::cout << "2x3 Matrix Singular Value Decomposition:"
            << std::endl
            << "  m2x3A matrix U: " << std::endl
            << m2x3Ajsvd.matrixU()
            << std::endl
            << "  m2x3A matrix V: " << std::endl
            << m2x3Ajsvd.matrixV()
            << std::endl << std::endl;

  Eigen::JacobiSVD< Eigen::Matrix<double,3,2> > m3x2Ajsvd(m3x2A, Eigen::ComputeFullU|Eigen::ComputeFullV);
  std::cout << "3x2 Matrix Singular Value Decomposition:"
            << std::endl
            << "  m3x2A matrix U: " << std::endl
            << m3x2Ajsvd.matrixU()
            << std::endl
            << "  m3x2A matrix V: " << std::endl
            << m3x2Ajsvd.matrixV()
            << std::endl << std::endl;

  // Moore-Penrose Pseudoinverse
  // http://eigen.tuxfamily.org/index.php?title=FAQ#Is_there_a_method_to_compute_the_.28Moore-Penrose.29_pseudo_inverse_.3F
  /*
  void pinv(MatrixType& pinvmat) const {
    eigen_assert(m_isInitialized && "SVD is not initialized.");
    double pinvtoler = 1.e-6; // choose your tolerance wisely!
    SingularValuesType singularValues_inv = m_singularValues;
    for(size_t i=0; i<m_workMatrix.cols(); i++) {
      if(m_singularValues(i) > pinvtoler) {
        singularValues_inv(i) = 1.0/m_singularValues(i);
      } else {
        singularValues_inv(i) = 0;
      }
    }
    pinvmat = (m_matrixV*singularValues_inv.asDiagonal()*m_matrixU.transpose());
  }
  */

  // Trace
  std::cout << "2D Vector Traces:"
            << std::endl
            << "  v2A: " << v2A.trace()
            << std::endl
            << "  v2B: " << v2B.trace()
            << std::endl
            << "  v2C: " << v2C.trace()
            << std::endl << std::endl;

  std::cout << "3D Vector Traces:"
            << std::endl
            << "  v3A: " << v3A.trace()
            << std::endl
            << "  v3B: " << v3B.trace()
            << std::endl
            << "  v3C: " << v3C.trace()
            << std::endl << std::endl;

  std::cout << "2x2 Matrix Traces:"
            << std::endl
            << "  m2x2A: " << m2x2A.trace()
            << std::endl
            << "  m2x2B: " << m2x2B.trace()
            << std::endl
            << "  m2x2C: " << m2x2C.trace()
            << std::endl << std::endl;

  std::cout << "2x3 Matrix Traces:"
            << std::endl
            << "  m2x3A: " << m2x3A.trace()
            << std::endl
            << "  m2x3B: " << m2x3B.trace()
            << std::endl
            << "  m2x3C: " << m2x3C.trace()
            << std::endl << std::endl;

  std::cout << "3x2 Matrix Traces:"
            << std::endl
            << "  m3x2A: " << m3x2A.trace()
            << std::endl
            << "  m3x2B: " << m3x2B.trace()
            << std::endl
            << "  m3x2C: " << m3x2C.trace()
            << std::endl << std::endl;

  std::cout << "3x3x3 Tensor Traces: TODO"
            << std::endl << std::endl;

  // Determinant
  std::cout << "2x2 Matrix Determinants:"
            << std::endl
            << "  m2x2A: " << m2x2A.determinant()
            << std::endl
            << "  m2x2B: " << m2x2B.determinant()
            << std::endl
            << "  m2x2C: " << m2x2C.determinant()
            << std::endl << std::endl;

  std::cout << "3x3x3 Tensor Determinants: TODO"
            << std::endl << std::endl;

  // Principal Components Analysis
  // later

  return 0;
}
