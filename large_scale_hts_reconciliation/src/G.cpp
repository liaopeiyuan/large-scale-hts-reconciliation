#include "G.h"
namespace lhts {
namespace G {

SpMat build_sparse_OLS(SpMat S) {
  SpMat St = S.transpose();
  SpMat M = St * S;
  SparseQR<SpMat, COLAMDOrdering<int>> solver;
  solver.compute(M);
  return solver.solve(St);
}

SpMat build_sparse_WLS(SpMat S, double w) {
  std::vector<T> tripletList(0);

  for (int i = 0; i < S.rows(); i++) {
    tripletList.push_back(T(i, i, w));
  }

  SpMat W(S.rows(), S.rows());
  W.setFromTriplets(tripletList.begin(), tripletList.end());

  SpMat St = S.transpose();
  SpMat M = St * W * S;

  SparseQR<SpMat, COLAMDOrdering<int>> solver;
  solver.compute(M);
  return solver.solve(St) * W;
}

SpMat build_sparse_top_down(const MatrixXi S_compact, const MatrixXd P,
                            int num_leaves, int num_nodes, int num_levels) {
  SpMat G(num_leaves, num_nodes);

  if (S_compact.rows() != num_nodes)
    throw std::invalid_argument("Hierarchy does not correspond to all nodes.");
  if (S_compact.cols() != num_levels)
    throw std::invalid_argument(
        "Hierarchy does not correspond to all nodes' path to root.");
  if (num_levels <= 1)
    throw std::invalid_argument("No hierarchy (num_levels <=1).");
  if (P.rows() != num_nodes)
    throw std::invalid_argument("P does not correspond to all nodes.");

  std::vector<T> tripletList;

  for (int i = 0; i < num_nodes; i++) {
    int co = S_compact(i, 0);
    int root = -1;
    bool is_base = true;
    for (int j = 1; j < num_levels; j++) {
      int ro = S_compact(i, j);
      if (ro == -1) {
        is_base = false;
        break;
      }
      root = ro;
    }
    if (is_base) {
      tripletList.push_back(T(co, root, P(co, 0)));
    }
  }

  G.setFromTriplets(tripletList.begin(), tripletList.end());

  return G;
}

SpMat build_sparse_middle_out(const MatrixXi S_compact, const MatrixXd P,
                              int level, int num_leaves, int num_nodes,
                              int num_levels) {
  SpMat G(num_leaves, num_nodes);

  if (S_compact.rows() != num_nodes)
    throw std::invalid_argument("Hierarchy does not correspond to all nodes.");
  if (S_compact.cols() != num_levels)
    throw std::invalid_argument(
        "Hierarchy does not correspond to all nodes' path to root.");
  if (num_levels <= 1)
    throw std::invalid_argument("No hierarchy (num_levels <=1).");
  if (P.rows() != num_nodes)
    throw std::invalid_argument("P does not correspond to all nodes.");
  if (level < 1 || level >= num_levels)
    throw std::invalid_argument("Invalid level.");

  std::vector<T> tripletList;

  for (int i = 0; i < num_nodes; i++) {
    int co = S_compact(i, 0);
    int root = co;
    int lvl = num_levels - level;
    bool is_base = true;
    for (int j = 1; j < num_levels; j++) {
      int ro = S_compact(i, j);
      if (ro == -1) {
        is_base = false;
        break;
      }
      if (lvl > 0) {
        root = ro;
        lvl--;
      }
    }
    if (is_base) {
      tripletList.push_back(T(co, root, P(co, 0)));
    }
  }

  G.setFromTriplets(tripletList.begin(), tripletList.end());

  return G;
}

SpMat build_sparse_bottom_up(const MatrixXi S_compact, int num_leaves,
                             int num_nodes, int num_levels) {
  SpMat G(num_leaves, num_nodes);

  if (S_compact.rows() != num_nodes)
    throw std::invalid_argument("Hierarchy does not correspond to all nodes.");
  if (S_compact.cols() != num_levels)
    throw std::invalid_argument(
        "Hierarchy does not correspond to all nodes' path to root.");
  if (num_levels <= 1)
    throw std::invalid_argument("No hierarchy (num_levels <=1).");

  std::vector<T> tripletList;

  for (int i = 0; i < num_nodes; i++) {
    int co = S_compact(i, 0);
    bool is_base = true;
    for (int j = 1; j < num_levels; j++) {
      int ro = S_compact(i, j);
      if (ro == -1) {
        is_base = false;
        break;
      }
    }
    if (is_base) {
      tripletList.push_back(T(co, i, 1.0));
    }
  }

  G.setFromTriplets(tripletList.begin(), tripletList.end());

  return G;
}

MatrixXd build_dense_OLS(const MatrixXi S) {
  MatrixXd Sp = S.cast<double>().eval();
  MatrixXd St = Sp.transpose().eval();
  MatrixXd M = St * Sp;
  HouseholderQR<MatrixXd> qr(M);
  return qr.solve(St);
}

MatrixXd build_dense_WLS(const MatrixXi S, double w) {
  MatrixXd W = MatrixXd::Zero(S.rows(), S.cols());
#pragma omp parallel for
  for (int i = 0; i < S.rows(); i++) {
    W(i, i) = w;
  }
  MatrixXd Sp = S.cast<double>();
  MatrixXd St = Sp.transpose();
  MatrixXd M = St * W * Sp;
  HouseholderQR<MatrixXd> qr(M);
  return qr.solve(St) * W;
}

MatrixXd build_dense_top_down(const MatrixXi S_compact, const MatrixXd P,
                              int num_leaves, int num_nodes, int num_levels) {
  MatrixXd G = MatrixXd::Zero(num_leaves, num_nodes);

  if (S_compact.rows() != num_nodes)
    throw std::invalid_argument("Hierarchy does not correspond to all nodes.");
  if (S_compact.cols() != num_levels)
    throw std::invalid_argument(
        "Hierarchy does not correspond to all nodes' path to root.");
  if (num_levels <= 1)
    throw std::invalid_argument("No hierarchy (num_levels <=1).");
  if (P.rows() != num_nodes)
    throw std::invalid_argument("P does not correspond to all nodes.");

#pragma omp parallel for
  for (int i = 0; i < num_nodes; i++) {
    int co = S_compact(i, 0);
    int root = -1;
    bool is_base = true;
    for (int j = 1; j < num_levels; j++) {
      int ro = S_compact(i, j);
      if (ro == -1) {
        is_base = false;
        break;
      }
      root = ro;
    }
    if (is_base) {
      G(co, root) = P(co, 0);
    }
  }

  return G;
}

MatrixXd build_dense_middle_out(const MatrixXi S_compact, const MatrixXd P,
                                int level, int num_leaves, int num_nodes,
                                int num_levels) {
  MatrixXd G = MatrixXd::Zero(num_leaves, num_nodes);

  if (S_compact.rows() != num_nodes)
    throw std::invalid_argument("Hierarchy does not correspond to all nodes.");
  if (S_compact.cols() != num_levels)
    throw std::invalid_argument(
        "Hierarchy does not correspond to all nodes' path to root.");
  if (num_levels <= 1)
    throw std::invalid_argument("No hierarchy (num_levels <=1).");
  if (P.rows() != num_nodes)
    throw std::invalid_argument("P does not correspond to all nodes.");
  if (level < 1 || level >= num_levels)
    throw std::invalid_argument("Invalid level.");

#pragma omp parallel for
  for (int i = 0; i < num_nodes; i++) {
    int co = S_compact(i, 0);
    int root = co;
    int lvl = num_levels - level;
    bool is_base = true;
    for (int j = 1; j < num_levels; j++) {
      int ro = S_compact(i, j);
      if (ro == -1) {
        is_base = false;
        break;
      }
      if (lvl > 0) {
        root = ro;
        lvl--;
      }
    }
    if (is_base) {
      G(co, root) = P(co, 0);
    }
  }

  return G;
}

MatrixXd build_dense_bottom_up(const MatrixXi S_compact, int num_leaves,
                               int num_nodes, int num_levels) {
  MatrixXd G = MatrixXd::Zero(num_leaves, num_nodes);

  if (S_compact.rows() != num_nodes)
    throw std::invalid_argument("Hierarchy does not correspond to all nodes.");
  if (S_compact.cols() != num_levels)
    throw std::invalid_argument(
        "Hierarchy does not correspond to all nodes' path to root.");
  if (num_levels <= 1)
    throw std::invalid_argument("No hierarchy (num_levels <=1).");

#pragma omp parallel for
  for (int i = 0; i < num_nodes; i++) {
    int co = S_compact(i, 0);
    bool is_base = true;
    for (int j = 1; j < num_levels; j++) {
      int ro = S_compact(i, j);
      if (ro == -1) {
        is_base = false;
        break;
      }
    }
    if (is_base) {
      G(i, i) = 1.0;
    }
  }

  return G;
}

}  // namespace G
}  // namespace lhts