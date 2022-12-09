
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

#include "G.h"
#include "S.h"
#include "distribute.h"

using namespace Eigen;

namespace lhts {
namespace reconcile {
MatrixXd sparse_matrix(const std::string method, const MatrixXi S_compact,
                      int num_leaves, int num_nodes, int num_levels, const MatrixXd yhat,
                       const MatrixXd P = MatrixXd(), int level = 2, double w = 1.0);

MatrixXd sparse_algo(const std::string method, const MatrixXi S_compact,
                      int num_leaves, int num_nodes, int num_levels, const MatrixXd yhat,
                       const MatrixXd P = MatrixXd(), int level = 2, double w = 1.0);

MatrixXd dense_matrix(const std::string method, const MatrixXi S_compact,
                      int num_leaves, int num_nodes, int num_levels, const MatrixXd yhat,
                       const MatrixXd P = MatrixXd(), int level = 2, double w = 1.0);

MatrixXd dense_algo(const std::string method, const MatrixXi S_compact,
                      int num_leaves, int num_nodes, int num_levels, const MatrixXd yhat,
                       const MatrixXd P = MatrixXd(), int level = 2, double w = 1.0);
}  // namespace reconcile
}  // namespace lhts