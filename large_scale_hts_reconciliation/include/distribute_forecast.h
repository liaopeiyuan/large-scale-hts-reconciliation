// Referencing https://github.com/latug0/pybind_mpi/

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>
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

typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpMat;
typedef Eigen::Triplet<double> T;

using namespace Eigen;

namespace lhts {
namespace distribute_forecast {
MatrixXd top_down(const MatrixXi S_compact, const MatrixXd P,
                  const MatrixXd yhat, int num_leaves, int num_nodes,
                  int num_levels);
MatrixXd middle_out(const MatrixXi S_compact, const MatrixXd P,
                    const MatrixXd yhat, int level, int num_leaves,
                    int num_nodes, int num_levels);
}  // namespace distribute_forecast
}  // namespace lhts