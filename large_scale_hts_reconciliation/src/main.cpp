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
#include "S.h"
#include "distribute.h"
#include "reconcile.h"
#include "MPI_utils.h"

using namespace lhts;
using namespace Eigen;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

typedef SparseMatrix<float, ColMajor> SpMat;
typedef Triplet<float> T;


float rmse(const MatrixXf res, const MatrixXf gt) {
  float sum = 0;
  for (int i = 0; i < res.rows(); i++) {
    for (int j = 0; j < res.cols(); j++) {
      sum += pow(abs(res(i, j) - gt(i, j)), 2);
    }
  }
  float rmse = sqrt(sum / (res.rows() * res.cols()));
  return rmse;
}

float mae(const MatrixXf res, const MatrixXf gt) {
  float sum = 0;
  for (int i = 0; i < res.rows(); i++) {
    for (int j = 0; j < res.cols(); j++) {
      sum += abs(res(i, j) - gt(i, j));
    }
  }
  float mae = sum / (res.rows() * res.cols());
  return mae;
}

float smape(const MatrixXf res, const MatrixXf gt) {
  float sum = 0;
  for (int i = 0; i < res.rows(); i++) {
    for (int j = 0; j < res.cols(); j++) {
      float pd = res(i, j);
      float gtr = gt(i, j);
      float abse = abs(pd - gtr);
      float mean = (abs(pd) + abs(gtr)) / 2;
      if (mean == 0) {
        sum += 0;
      } else {
        float val = abse / mean;
        sum += val;
      }
    }
  }
  float smape = sum / (res.rows() * res.cols());
  return smape;
}

namespace py = pybind11;
using pymod = pybind11::module;


PYBIND11_MODULE(lhts, m) {
  m.doc() = R"pbdoc(
        LHTS
        -----------------------
        .. currentmodule:: lhts
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

  m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");

  m.def("rmse", &rmse);
  m.def("mae", &mae);
  m.def("smape", &smape);

  m.def("reconcile_matrix", &reconcile::reconcile_matrix);
  m.def("reconcile", &reconcile::reconcile);
  m.def("construct_S", &S::build_sparse);
  m.def("G::build_sparse_bottom_up", &G::build_sparse_bottom_up);
  m.def("G::build_sparse_top_down", &G::build_sparse_top_down);
  m.def("G::build_sparse_middle_out", &G::build_sparse_middle_out);

  py::class_<MPI_utils>(m, "MPI_utils")
      .def(py::init<>())
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