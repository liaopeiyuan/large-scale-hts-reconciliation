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
#include<Eigen/SparseCholesky>

typedef Eigen::SparseMatrix<float, Eigen::ColMajor> SpMat;
typedef Eigen::Triplet<float> T;

using namespace Eigen;
namespace lhts {
namespace G {
SpMat build_sparse_OLS(SpMat S);
SpMat build_sparse_WLS(SpMat S, float w);
SpMat build_sparse_top_down(const MatrixXi S_compact, const MatrixXf P,
                            int num_leaves, int num_nodes, int num_levels);
SpMat build_sparse_middle_out(const MatrixXi S_compact, const MatrixXf P,
                              int level, int num_leaves, int num_nodes,
                              int num_levels);

SpMat build_sparse_bottom_up(const MatrixXi S_compact, int num_leaves,
                             int num_nodes, int num_levels);
}  // namespace G
}  // namespace lhts