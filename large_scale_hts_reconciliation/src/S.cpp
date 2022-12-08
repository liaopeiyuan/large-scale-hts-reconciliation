#include "S.h"

namespace lhts {
namespace S {
SpMat build_sparse(const MatrixXi S_compact, int num_leaves, int num_nodes,
                   int num_levels) {
  SpMat S(num_nodes, num_leaves);

  std::vector<T> tripletList;

  if (S_compact.rows() != num_nodes)
    throw std::invalid_argument("Hierarchy does not correspond to all nodes.");
  if (S_compact.cols() != num_levels)
    throw std::invalid_argument(
        "Hierarchy does not correspond to all nodes' path to root.");
  if (num_levels <= 1)
    throw std::invalid_argument("No hierarchy (num_levels <=1).");

  for (int i = 0; i < num_leaves; i++) {
    int co = S_compact(i, 0);
    if (co >= num_leaves) {
      throw std::invalid_argument(
          "Make sure that the frist num_leaves rows of "
          "S_compact contain only leaf-level nodes.");
    }
    tripletList.push_back(T(co, co, 1));
    for (int j = 1; j < num_levels; j++) {
      int ro = S_compact(i, j);
      if (ro == -1) {
        if (i < num_leaves) {
          throw std::invalid_argument(
              "Make sure that the frist num_leaves rows of S_compact contain "
              "only leaf-level nodes.");
        }
        break;
      } else {
        if (co >= num_leaves) {
          throw std::invalid_argument(
              "Make sure that the all leaf-level nodes have index < "
              "num_leaves.");
        }
        tripletList.push_back(T(ro, co, 1));
      }
    }
  }

  S.setFromTriplets(tripletList.begin(), tripletList.end());

  return S;
}

MatrixXi build_dense(const MatrixXi S_compact, int num_leaves, int num_nodes,
                     int num_levels) {
  MatrixXi S = MatrixXi::Zero(num_nodes, num_leaves);

  if (S_compact.rows() != num_nodes)
    throw std::invalid_argument("Hierarchy does not correspond to all nodes.");
  if (S_compact.cols() != num_levels)
    throw std::invalid_argument(
        "Hierarchy does not correspond to all nodes' path to root.");
  if (num_levels <= 1)
    throw std::invalid_argument("No hierarchy (num_levels <=1).");

#pragma omp parallel for
  for (int i = 0; i < num_leaves; i++) {
    int co = S_compact(i, 0);
    if (co >= num_leaves) {
      throw std::invalid_argument(
          "Make sure that the frist num_leaves rows of "
          "S_compact contain only leaf-level nodes.");
    }
    S(co, co) = 1;
    for (int j = 1; j < num_levels; j++) {
      int ro = S_compact(i, j);
      if (ro == -1) {
        if (i < num_leaves) {
          throw std::invalid_argument(
              "Make sure that the frist num_leaves rows of S_compact contain "
              "only leaf-level nodes.");
        }
        break;
      } else {
        if (co >= num_leaves) {
          throw std::invalid_argument(
              "Make sure that the all leaf-level nodes have index < "
              "num_leaves.");
        }
        S(ro, co) = 1;
      }
    }
  }

  return S;
}
}  // namespace S
}  // namespace lhts