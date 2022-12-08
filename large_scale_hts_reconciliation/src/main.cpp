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

#include "G.h"
#include "MPI_utils.h"
#include "S.h"
#include "distribute.h"
#include "metrics.h"
#include "reconcile.h"

using namespace lhts;
using namespace Eigen;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

typedef SparseMatrix<float, ColMajor> SpMat;
typedef Triplet<float> T;

namespace py = pybind11;
using pymod = pybind11::module;

PYBIND11_MODULE(lhts, m) {
  m.doc() = R"pbdoc(
        LHTS
        -----------------------
        .. currentmodule:: lhts
        .. autosummary::
           :toctree: _generate
           reconcile
    )pbdoc";

  m.def("rmse", &metrics::rmse);
  m.def("mae", &metrics::mae);
  m.def("smape", &metrics::smape);

  m.def("reconcile_sparse_matrix", &reconcile::sparse_matrix);
  m.def("reconcile_sparse_algo", &reconcile::sparse_algo, R"pbdoc(
        Forecast reconciliation
    )pbdoc");
  m.def("build_S_sparse", &S::build_sparse);
  m.def("build_G_sparse_bottom_up", &G::build_sparse_bottom_up);
  m.def("build_G_sparse_top_down", &G::build_sparse_top_down);
  m.def("build_G_sparse_middle_out", &G::build_sparse_middle_out);

  py::class_<MPI_utils>(m, "MPI_utils")
      .def(py::init<>())
      .def("test_mpi", &MPI_utils::test, "test")
      .def("reconcile_gather", &MPI_utils::reconcile_gather, "reconcile_gather")
      .def("reconcile_dp_matrix", &MPI_utils::reconcile_dp_matrix,
           "reconcile_dp_matrix")
      .def("reconcile_dp_optimized", &MPI_utils::reconcile_dp_optimized,
           "reconcile_dp_matrix");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}