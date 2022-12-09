#include "distribute_forecast.h"

namespace lhts {
namespace distribute_forecast {
MatrixXd top_down(const MatrixXi S_compact, const MatrixXd P,
                  const MatrixXd yhat, int num_leaves, int num_nodes,
                  int num_levels) {
  MatrixXd y = MatrixXd::Zero(num_leaves, yhat.cols());

  if (S_compact.rows() != num_nodes)
    throw std::invalid_argument("Hierarchy does not correspond to all nodes.");
  if (S_compact.cols() != num_levels)
    throw std::invalid_argument(
        "Hierarchy does not correspond to all nodes' path to root.");
  if (num_levels <= 1)
    throw std::invalid_argument("No hierarchy (num_levels <=1).");
  if (P.cols() > 1)
    std::cerr << "[lhts] Warning: Only the first column of the proportions "
                 "matrix will be used\n";

#pragma omp simd
  for (int i = 0; i < num_leaves; i++) {
    int co = S_compact(i, 0);
    int root = S_compact(i, num_levels - 1);
    int is_base = root != -1;
    if (is_base) {
      y.middleRows(co, 1) = P(co, 0) * yhat.middleRows(root, 1);
    }
  }

  return y;
}

MatrixXd middle_out(const MatrixXi S_compact, const MatrixXd P,
                    const MatrixXd yhat, int level, int num_leaves,
                    int num_nodes, int num_levels) {
  MatrixXd y = MatrixXd::Zero(num_leaves, yhat.cols());

  if (S_compact.rows() != num_nodes)
    throw std::invalid_argument("Hierarchy does not correspond to all nodes.");
  if (S_compact.cols() != num_levels)
    throw std::invalid_argument(
        "Hierarchy does not correspond to all nodes' path to root.");
  if (num_levels <= 1)
    throw std::invalid_argument("No hierarchy (num_levels <=1).");
  if (P.cols() > 1)
    std::cerr << "[lhts] Warning: Only the first column of the proportions "
                 "matrix will be used\n";

#pragma omp simd
  for (int i = 0; i < num_leaves; i++) {
    int co = S_compact(i, 0);
    int lvl = num_levels - level;
    int root = S_compact(i, lvl);
    bool is_base = S_compact(i, num_levels - 1) != -1;
    if (is_base) {
      y.middleRows(co, 1) = P(co, 0) * yhat.middleRows(root, 1);
    }
  }

  return y;
}
}  // namespace distribute_forecast
}  // namespace lhts