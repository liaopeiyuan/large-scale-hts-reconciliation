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


namespace lhts {
    namespace distribute {
MatrixXf top_down(const MatrixXi S_compact,
                                      const MatrixXf P, const MatrixXf yhat,
                                      int num_base, int num_total,
                                      int num_levels);
MatrixXf middle_out(const MatrixXi S_compact,
                                        const MatrixXf P, const MatrixXf yhat,
                                        int level, int num_base, int num_total,
                                        int num_levels);
    }
}