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


using namespace Eigen;

namespace lhts {
namespace metrics {
float rmse(const MatrixXf res, const MatrixXf gt);

float mae(const MatrixXf res, const MatrixXf gt);

float smape(const MatrixXf res, const MatrixXf gt);

}  // namespace reconcile
}  // namespace lhts