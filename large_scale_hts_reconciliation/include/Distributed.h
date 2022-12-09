// Referencing https://github.com/latug0/pybind_mpi/

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
#include "reconcile.h"

using namespace Eigen;

typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpMat;
typedef Eigen::Triplet<double> T;

namespace lhts {
class Distributed {
 public:
  Distributed() : comm_global(MPI_COMM_WORLD) { initParallel(); }

  ~Distributed() {}

  MatrixXd reconcile_dp_optimized(const std::string method, const MatrixXi S_compact,
                      int num_leaves, int num_nodes, int num_levels, const MatrixXd yhat,
                       const MatrixXd P, int level, double w);

  MatrixXd reconcile_dp_matrix(const std::string method, const MatrixXi S_compact,
                      int num_leaves, int num_nodes, int num_levels, const MatrixXd yhat,
                       const MatrixXd P, int level, double w);

  MatrixXd reconcile_gather(const std::string method, const MatrixXi S_compact,
                      int num_leaves, int num_nodes, int num_levels, const MatrixXd yhat,
                       const MatrixXd P, int level, double w);

  void test(const MatrixXd& xs);

 private:
  MPI_Comm comm_global;
};
}  // namespace lhts