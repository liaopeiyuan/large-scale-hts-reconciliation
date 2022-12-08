#include "reconcile.h"

namespace lhts {
namespace reconcile {
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

MatrixXf reconcile(const std::string method, const MatrixXi S_compact,
                   const MatrixXf P, const MatrixXf yhat, int level, float w,
                   int num_base, int num_total, int num_levels) {
  SpMat S = S::build_sparse(S_compact, num_base, num_total, num_levels);

  // std::stringstream ss;
  // ss << S.rows() << " " << S.cols() << " " << S(seqN(0, 10), seqN(0, 10));
  // printf("S: %s\n", ss.str().c_str());

  SpMat G;
  MatrixXf result, y;
  y = yhat;

  if (method == "bottom_up") {
    y = yhat.topRows(num_base).eval();
    result = S * y;
  } else if (method == "top_down") {
    y = distribute::top_down(S_compact, P, yhat, num_base, num_total,
                             num_levels);
    result = S * y;
  } else if (method == "middle_out") {
    y = distribute::middle_out(S_compact, P, yhat, level, num_base, num_total,
                               num_levels);
    result = S * y;
  } else if (method == "OLS") {
    G = G::build_sparse_OLS(S);
    result = (S * G) * y;
  } else if (method == "WLS") {
    G = G::build_sparse_WLS(S, w);
    res = S * G;
  } else {
    throw std::invalid_argument(
        "invalid reconciliation method. Available options are: bottom_up, "
        "top_down, middle_out, OLS, WLS");
  }

  return result;
}
}  // namespace reconcile
}  // namespace lhts