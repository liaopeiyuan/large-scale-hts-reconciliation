#include "reconcile.h"

#include <chrono>
class Timer {
public:
  Timer() : beg_(clock_::now()) {}
  void reset() { beg_ = clock_::now(); }
  double elapsed() const {
    return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
  }

private:
  typedef std::chrono::high_resolution_clock clock_;
  typedef std::chrono::duration<double, std::ratio<1>> second_;
  std::chrono::time_point<clock_> beg_;
};

namespace lhts {
namespace reconcile {

MatrixXd sparse_matrix(const std::string method, const MatrixXi S_compact,
                      int num_leaves, int num_nodes, int num_levels, const MatrixXd yhat,
                       const MatrixXd P, int level, double w) {
  SpMat S = S::build_sparse(S_compact, num_leaves, num_nodes, num_levels);
  SpMat G, res;
  MatrixXd result, y;
  y = yhat;

  Timer totalSimulationTimer;
  double totalSimulationTime;

  if (method == "bottom_up") {
    G = G::build_sparse_bottom_up(S_compact, num_leaves, num_nodes, num_levels);
    res = S * G;
  } else if (method == "top_down") {
    G = G::build_sparse_top_down(S_compact, P, num_leaves, num_nodes,
                                 num_levels);
    res = S * G;
  } else if (method == "middle_out") {
    G = G::build_sparse_middle_out(S_compact, P, level, num_leaves, num_nodes,
                                   num_levels);
    totalSimulationTime = totalSimulationTimer.elapsed();
    printf("total simulation time: %.6fs\n", totalSimulationTime);
    res = S * G;
    totalSimulationTime = totalSimulationTimer.elapsed();
    printf("total simulation time: %.6fs\n", totalSimulationTime);
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
  totalSimulationTime = totalSimulationTimer.elapsed();
  printf("total simulation time: %.6fs\n", totalSimulationTime);

  return result;
}

MatrixXd dense_matrix(const std::string method, const MatrixXi S_compact,
                      int num_leaves, int num_nodes, int num_levels, const MatrixXd yhat,
                       const MatrixXd P, int level, double w) {
  MatrixXi S = S::build_dense(S_compact, num_leaves, num_nodes, num_levels);

  MatrixXd G, res, y;
  y = yhat;

  if (method == "bottom_up") {
    G = G::build_dense_bottom_up(S_compact, num_leaves, num_nodes, num_levels);
    res = S.cast<double>() * G;
  } else if (method == "top_down") {
    G = G::build_dense_top_down(S_compact, P, num_leaves, num_nodes,
                                num_levels);
    res = S.cast<double>() * G;
  } else if (method == "middle_out") {
    G = G::build_dense_middle_out(S_compact, P, level, num_leaves, num_nodes,
                                  num_levels);
    res = S.cast<double>() * G;
  } else if (method == "OLS") {
    G = G::build_dense_OLS(S);
    res = S.cast<double>() * G;
  } else if (method == "WLS") {
    G = G::build_dense_WLS(S, w);
    res = S.cast<double>() * G;
  } else {
    throw std::invalid_argument(
        "invalid reconciliation method. Available options are: bottom_up, "
        "top_down, middle_out, OLS, WLS");
  }

  res = res * y;

  return res;
}

MatrixXd sparse_algo(const std::string method, const MatrixXi S_compact,
                      int num_leaves, int num_nodes, int num_levels, const MatrixXd yhat,
                       const MatrixXd P, int level, double w) {
  SpMat S = S::build_sparse(S_compact, num_leaves, num_nodes, num_levels);

  SpMat G;
  MatrixXd result, y;
  y = yhat;

  Timer totalSimulationTimer;
  double totalSimulationTime;
  
  if (method == "bottom_up") {
    y = yhat.topRows(num_leaves).eval();
    result = yhat;
    result.bottomRows(num_nodes - num_leaves) = S.bottomRows(num_nodes - num_leaves) * y;
  } else if (method == "top_down") {
    y = distribute::top_down(S_compact, P, yhat, num_leaves, num_nodes,
                             num_levels);
    result = S * y;
  } else if (method == "middle_out") {
    y = distribute::middle_out(S_compact, P, yhat, level, num_leaves, num_nodes,
                               num_levels);
    totalSimulationTime = totalSimulationTimer.elapsed();
    printf("total simulation time: %.6fs\n", totalSimulationTime);
    result = S * y;
    totalSimulationTime = totalSimulationTimer.elapsed();
    printf("total simulation time: %.6fs\n", totalSimulationTime);
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

MatrixXd dense_algo(const std::string method, const MatrixXi S_compact,
                      int num_leaves, int num_nodes, int num_levels, const MatrixXd yhat,
                       const MatrixXd P, int level, double w) {
  MatrixXi S = S::build_dense(S_compact, num_leaves, num_nodes, num_levels);

  MatrixXd G, res, y;
  y = yhat;

  if (method == "bottom_up") {
    res = S.cast<double>();
    y = yhat.topRows(num_leaves).eval();
  } else if (method == "top_down") {
    res = S.cast<double>();
    y = distribute::top_down(S_compact, P, yhat, num_leaves, num_nodes,
                             num_levels);
  } else if (method == "middle_out") {
    res = S.cast<double>();
    y = distribute::middle_out(S_compact, P, yhat, level, num_leaves, num_nodes,
                               num_levels);
  } else if (method == "OLS") {
    G = G::build_dense_OLS(S);
    res = S.cast<double>() * G;
  } else if (method == "WLS") {
    G = G::build_dense_WLS(S, w);
    res = S.cast<double>() * G;
  } else {
    throw std::invalid_argument(
        "invalid reconciliation method. Available options are: bottom_up, "
        "top_down, middle_out, OLS, WLS");
  }

  res = res * y;

  return res;
}

}  // namespace reconcile
}  // namespace lhts