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
#include "MPI_utils.h"

using namespace lhts;
using namespace Eigen;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

typedef SparseMatrix<float, ColMajor> SpMat;
typedef Triplet<float> T;

MatrixXf reconcile_matrix(const std::string method, const MatrixXi S_compact,
                          const MatrixXf P, const MatrixXf yhat, int level,
                          float w, int num_base, int num_total,
                          int num_levels) {
  SpMat S = S::build_sparse(S_compact, num_base, num_total, num_levels);
  SpMat G, res;
  MatrixXf result, y;
  y = yhat;

  if (method == "bottom_up") {
    G = G::build_sparse_bottom_up(S_compact, num_base, num_total, num_levels);
    res = S * G;
  } else if (method == "top_down") {
    G = G::build_sparse_top_down(S_compact, P, num_base, num_total, num_levels);
    res = S * G;
  } else if (method == "middle_out") {
    G = G::build_sparse_middle_out(S_compact, P, level, num_base, num_total,
                                   num_levels);
    res = S * G;
  } else if (method == "OLS") {
    G = G::build_sparse_OLS(S).sparseView();
    res = S * G;
  }
  /*
else if (method == "WLS") {
    G = G::build_sparse_WLS(S, w);
    res = S * G;
} */
  else {
    throw std::invalid_argument(
        "invalid reconciliation method. Available options are: bottom_up, "
        "top_down, middle_out, OLS, WLS");
  }

  result = res * y;

  return result;
}

MatrixXf reconcile(const std::string method, const MatrixXi S_compact,
                   const MatrixXf P, const MatrixXf yhat, int level, float w,
                   int num_base, int num_total, int num_levels) {
  SpMat S = S::build_sparse(S_compact, num_base, num_total, num_levels);

  // std::stringstream ss;
  // ss << S.rows() << " " << S.cols() << " " << S(seqN(0, 10), seqN(0, 10));
  // printf("S: %s\n", ss.str().c_str());

  SpMat G, res;
  MatrixXf result, y;
  y = yhat;

  if (method == "bottom_up") {
    res = S;
    y = yhat.topRows(num_base).eval();
  } else if (method == "top_down") {
    res = S;
    y = distribute::top_down(S_compact, P, yhat, num_base, num_total,
                             num_levels);
  } else if (method == "middle_out") {
    res = S;
    y = distribute::middle_out(S_compact, P, yhat, level, num_base, num_total,
                               num_levels);
  } else if (method == "OLS") {
    G = G::build_sparse_OLS(S).sparseView();
    res = S * G;
  }
  /*
else if (method == "WLS") {
    G = G::build_sparse_WLS(S, w);
    res = S * G;
} */
  else {
    throw std::invalid_argument(
        "invalid reconciliation method. Available options are: bottom_up, "
        "top_down, middle_out, OLS, WLS");
  }

  result = res * y;

  return result;
}

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

  m.def("reconcile_matrix", &reconcile_matrix);
  m.def("reconcile", &reconcile);
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