#include "reconcile.h"

namespace lhts {
namespace reconcile {

MatrixXf sparse_matrix(const std::string method, const MatrixXi S_compact,
                          const MatrixXf P, const MatrixXf yhat, int level,
                          float w, int num_leaves, int num_nodes,
                          int num_levels) {
  SpMat S = S::build_sparse(S_compact, num_leaves, num_nodes, num_levels);
  SpMat G, res;
  MatrixXf result, y;
  y = yhat;

  if (method == "bottom_up") {
    G = G::build_sparse_bottom_up(S_compact, num_leaves, num_nodes, num_levels);
    res = S * G;
  } else if (method == "top_down") {
    G = G::build_sparse_top_down(S_compact, P, num_leaves, num_nodes, num_levels);
    res = S * G;
  } else if (method == "middle_out") {
    G = G::build_sparse_middle_out(S_compact, P, level, num_leaves, num_nodes,
                                   num_levels);
    res = S * G;
  } else if (method == "OLS") {
    G = G::build_sparse_OLS(S);
    res = S * G;
  } else if (method == "WLS") {
    G = G::build_sparse_WLS(S, w);
    res = S * G;
  } else {
    throw std::invalid_argument(
        "invalid reconciliation method. Available options are: bottom_up, "
        "top_down, middle_out, OLS, WLS");
  }

  result = res * y;

  return result;
}

MatrixXf sparse_algo(const std::string method, const MatrixXi S_compact,
                   const MatrixXf P, const MatrixXf yhat, int level, float w,
                   int num_leaves, int num_nodes, int num_levels) {
  SpMat S = S::build_sparse(S_compact, num_leaves, num_nodes, num_levels);

  SpMat G;
  MatrixXf result, y;
  y = yhat;

  if (method == "bottom_up") {
    y = yhat.topRows(num_leaves).eval();
    result = S * y;
  } else if (method == "top_down") {
    y = distribute::top_down(S_compact, P, yhat, num_leaves, num_nodes,
                             num_levels);
    result = S * y;
  } else if (method == "middle_out") {
    y = distribute::middle_out(S_compact, P, yhat, level, num_leaves, num_nodes,
                               num_levels);
    result = S * y;
  } else if (method == "OLS") {
    G = G::build_sparse_OLS(S);
    result = (S * G) * y;
  } else if (method == "WLS") {
    G = G::build_sparse_WLS(S, w);
    result = (S * G) * y;
  } else {
    throw std::invalid_argument(
        "invalid reconciliation method. Available options are: bottom_up, "
        "top_down, middle_out, OLS, WLS");
  }

  return result;
}
}  // namespace reconcile
}  // namespace lhts