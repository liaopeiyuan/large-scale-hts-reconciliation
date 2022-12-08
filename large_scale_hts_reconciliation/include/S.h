#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <vector>
#include <set>
#include <tuple>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>
#include <stdio.h>
#include <Eigen/LU>
#include <Eigen/Sparse>
#include<Eigen/SparseQR>
#include<Eigen/SparseLU>

typedef Eigen::SparseMatrix<float, Eigen::ColMajor> SpMat;
typedef Eigen::Triplet<float> T;

using namespace Eigen;
namespace lhts {
    namespace S {
        SpMat build_sparse(const Eigen::MatrixXi S_compact, int num_base, int num_total, int num_levels);
    }
}