
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <iostream>

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
                          const MatrixXd P, const MatrixXd yhat, int level,
                          double w, int num_leaves, int num_nodes, int num_levels);

MatrixXd sparse_algo(const std::string method, const MatrixXi S_compact,
                   const MatrixXd P, const MatrixXd yhat, int level, double w,
                   int num_leaves, int num_nodes, int num_levels);
}  // namespace reconcile
}  // namespace lhts