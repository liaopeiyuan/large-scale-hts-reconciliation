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
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpMat;
typedef Eigen::Triplet<double> T;

using namespace Eigen;
namespace lhts {
namespace G {
SpMat build_sparse_OLS(SpMat S);
SpMat build_sparse_WLS(SpMat S, double w);
SpMat build_sparse_top_down(const MatrixXi S_compact, const MatrixXd P,
                            int num_leaves, int num_nodes, int num_levels);
SpMat build_sparse_middle_out(const MatrixXi S_compact, const MatrixXd P,
                              int level, int num_leaves, int num_nodes,
                              int num_levels);

SpMat build_sparse_bottom_up(const MatrixXi S_compact, int num_leaves,
                             int num_nodes, int num_levels);
MatrixXd build_dense_OLS(const MatrixXi S);
MatrixXd build_dense_WLS(const MatrixXi S, double w);
MatrixXd build_dense_top_down(const MatrixXi S_compact,
                                     const MatrixXd P, int num_leaves,
                                     int num_nodes, int num_levels);
MatrixXd build_dense_middle_out(const MatrixXi S_compact,
                                       const MatrixXd P, int level,
                                       int num_leaves, int num_nodes,
                                       int num_levels);
MatrixXd build_dense_bottom_up(const MatrixXi S_compact,
                                      int num_leaves, int num_nodes,
                                      int num_levels);
}  // namespace G
}  // namespace lhts