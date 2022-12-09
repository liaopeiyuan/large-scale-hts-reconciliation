#include "distribute.h"

namespace lhts {
namespace distribute {
MatrixXd top_down(const MatrixXi S_compact, const MatrixXd P,
                  MatrixXd yhat, int num_leaves, int num_nodes,
                  int num_levels) {
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

#pragma omp parallel for
  for (int i = 0; i < num_leaves; i++) {
    int co = S_compact(i, 0);
    int root = S_compact(i, num_levels - 1);
    int is_base = root != -1;
    /*
    int root = -1;
    bool is_base = true;
    for (int j = 1; j < num_levels; j++) {
      int ro = S_compact(i, j);
      if (ro == -1) {
        is_base = false;
        break;
      }
      root = ro;
    }*/
    if (is_base) {
      yhat.middleRows(co, 1) = P(co, 0) * yhat.middleRows(root, 1);
    }
  }

  return yhat;
}

MatrixXd middle_out(const MatrixXi S_compact, const MatrixXd P,
                    MatrixXd yhat, int level, int num_leaves,
                    int num_nodes, int num_levels) {
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
  if (level < 1 || level >= num_levels)
    throw std::invalid_argument("Invalid level.");


#pragma omp parallel for
  for (int i = 0; i < num_leaves; i++) {
    int co = S_compact(i, 0);
    //int root = co;
    int lvl = num_levels - level;
    // bool is_base = true;
    int root = 1 + lvl;
    bool is_base = root != -1;
    /*
    for (int j = 1; j < num_levels; j++) {
      int ro = S_compact(i, j);
      if (ro == -1) {
        is_base = false;
        break;
      }
      if (lvl > 0) {
        root = ro;
        lvl--;
      } else {
        break;
      }
    }
    */
    if (is_base) {
      yhat.middleRows(co, 1) = P(co, 0) * yhat.middleRows(root, 1);
    }
  }

  return yhat;
}
}  // namespace distribute
}  // namespace lhts