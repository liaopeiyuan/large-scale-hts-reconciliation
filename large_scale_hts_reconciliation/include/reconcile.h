
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
        MatrixXf reconcile_matrix(const std::string method, const MatrixXi S_compact,
                          const MatrixXf P, const MatrixXf yhat, int level,
                          float w, int num_base, int num_total,
                          int num_levels);
        
        MatrixXf reconcile(const std::string method, const MatrixXi S_compact,
                   const MatrixXf P, const MatrixXf yhat, int level, float w,
                   int num_base, int num_total, int num_levels);
    }
}