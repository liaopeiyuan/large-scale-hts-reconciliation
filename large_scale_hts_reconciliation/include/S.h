#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>
#include <stdio.h>
#include <Eigen/LU>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>

typedef Eigen::SparseMatrix<float, Eigen::ColMajor> SpMat;
typedef Eigen::Triplet<float> T;

using namespace Eigen;
namespace lhts {
namespace S {
SpMat build_sparse(const MatrixXi S_compact, int num_leaves, int num_nodes,
                   int num_levels);
MatrixXi build_dense(const MatrixXi S_compact, int num_leaves, int num_nodes,
                   int num_levels);
}
}  // namespace lhts