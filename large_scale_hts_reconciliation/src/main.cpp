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

using namespace lhts;
using namespace Eigen;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

typedef SparseMatrix<float, ColMajor> SpMat;
typedef Triplet<float> T;

MatrixXf dp_reconcile_optimized(const std::string method,
                                const MatrixXi S_compact, const MatrixXf P,
                                const MatrixXf yhat, int level, float w,
                                int num_base, int num_total, int num_levels,
                                int slice_start, int slice_length) {
  SpMat S = S::build_sparse(S_compact, num_base, num_total, num_levels)
                .middleRows(slice_start, slice_length)
                .eval();
  SpMat res, G;

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
    y = distribute::middle_out(S_compact, P, yhat, level, num_base,
                                       num_total, num_levels);
  } else if (method == "OLS") {
    G = G::build_sparse_OLS(S).sparseView();
    res = S * G;
  }
  /*
else if (method == "WLS") {
    G = G::build_sparse_WLS(S, w);
    res = S.cast<float>() * G;
} */
  else {
    throw std::invalid_argument(
        "invalid reconciliation method. Available options are: bottom_up, "
        "top_down, middle_out, OLS, WLS");
  }

  result = res * y;

  return result;
}

SpMat construct_dp_reconciliation_matrix(const std::string method,
                                         const MatrixXi S_compact,
                                         const MatrixXf P, int level, float w,
                                         int num_base, int num_total,
                                         int num_levels, int slice_start,
                                         int slice_length) {
  SpMat S = S::build_sparse(S_compact, num_base, num_total, num_levels);

  SpMat G;

  if (method == "bottom_up") {
    G = G::build_sparse_bottom_up(S_compact, num_base, num_total, num_levels);
  } else if (method == "top_down") {
    G = G::build_sparse_top_down(S_compact, P, num_base, num_total, num_levels);
  } else if (method == "middle_out") {
    G = G::build_sparse_middle_out(S_compact, P, level, num_base, num_total,
                                   num_levels);
  } else if (method == "OLS") {
    G = G::build_sparse_OLS(S).sparseView();
  }
  /*
else if (method == "WLS") {
    G = G::build_sparse_WLS(S, w);
} */
  else {
    throw std::invalid_argument(
        "invalid reconciliation method. Available options are: bottom_up, "
        "top_down, middle_out, OLS, WLS");
  }

  SpMat S_slice = S.middleRows(slice_start, slice_length).eval();
  SpMat res = (S_slice * G).eval();

  return res;
}

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
    y = distribute::middle_out(S_compact, P, yhat, level, num_base,
                                       num_total, num_levels);
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

class MPI_Utils {
 public:
  MPI_Utils() : comm_global(MPI_COMM_WORLD) { initParallel(); }

  ~MPI_Utils() {}

  MatrixXf reconcile_dp_optimized(const std::string method,
                                  const MatrixXi S_compact, const MatrixXf P,
                                  const MatrixXf yhat, int level, float w,
                                  int num_base, int num_total, int num_levels) {
    int world_size;
    MPI_Comm_size(comm_global, &world_size);
    int world_rank;
    MPI_Comm_rank(comm_global, &world_rank);

    int ro = yhat.rows();
    int co = yhat.cols();

    std::vector<int> rows(world_size);
    std::vector<int> cols(world_size);

    std::vector<MPI_Request> reqs(world_size);
    std::vector<MPI_Status> stats(world_size);

    MPI_Allgather(&ro, 1, MPI_INT, rows.data(), 1, MPI_INT, comm_global);
    MPI_Allgather(&co, 1, MPI_INT, cols.data(), 1, MPI_INT, comm_global);

    if (world_rank == 0) {
      int n_cols = cols[0];
      for (int i = 1; i < world_size; i++) {
        if (cols[i] != n_cols) {
          char buffer[200];
          sprintf(buffer, "Error: cols[%d] != cols[0]\n", i);
          throw std::invalid_argument(buffer);
        }
      }
    }

    MatrixXf result;

    if (method == "bottom_up") {
      int slice_start = 0, slice_length = 0;
      int curr_row = 0;

      for (int i = 0; i < world_size; i++) {
        if (i == world_rank) {
          slice_start = curr_row;
          slice_length = rows[i];
          break;
        }

        curr_row += rows[i];
      }

      MatrixXf yhat_total =
          MatrixXf::Zero(std::accumulate(rows.begin(), rows.end(), 0), cols[0]);
      std::vector<MatrixXf> yhats(world_size);

      curr_row = 0;
      for (int i = 0; i < world_size; i++) {
        MPI_Comm leaf_comm;

        int color =
            (i == world_rank) | (slice_start + slice_length >= num_base);
        MPI_Comm_split(comm_global, color,
                       (i == world_rank) ? 0 : world_rank + world_size,
                       &leaf_comm);

        if (color == 1) {
          if (i != world_rank) {
            yhats[i] = MatrixXf::Zero(rows[i], cols[i]);
          } else {
            yhats[i] = yhat;
          }
          // printf("rank %d @ %d, %d-%d\n", world_rank, i, slice_start,
          // slice_length);

          MPI_Bcast(yhats[i].data(), rows[i] * cols[i], MPI_FLOAT, 0,
                    leaf_comm);
          yhat_total.middleRows(curr_row, rows[i]) = yhats[i].eval();

          MPI_Comm_free(&leaf_comm);
        }

        curr_row += rows[i];
      }

      if (slice_start + slice_length >= num_base) {
        result = dp_reconcile_optimized(method, S_compact, P, yhat_total, level,
                                        w, num_base, num_total, num_levels,
                                        slice_start, slice_length);
      } else {
        result = yhat.eval();
      }

    } else if (method == "top_down") {
      std::vector<int> slice_starts(world_size);
      std::vector<std::set<int>> recvs(world_size, std::set<int>());
      std::vector<std::set<int>> sends(world_size, std::set<int>());

      int curr_row = 0;

      for (int i = 0; i < world_size; i++) {
        slice_starts[i] = curr_row;
        curr_row += rows[i];
      }

      /*
if (world_rank == 0) {
    for (int i = 0; i < world_size; i++) {
        printf("%d \n", slice_starts[i]);
    }
}
*/
      std::vector<std::tuple<int, int, int>> root_triplets(0);

      for (int i = 0; i < num_base; i++) {
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
          int root_process = 0, leaf_process = 0;
          // if (world_rank == 0) printf("%d %d\n", root, co);
          for (int j = 0; j < world_size; j++) {
            if (slice_starts[j] + rows[j] > root) {
              root_process = j;
              break;
            }
          }

          for (int j = 0; j < world_size; j++) {
            if (slice_starts[j] + rows[j] > co) {
              leaf_process = j;
              break;
            }
          }

          if (leaf_process == world_rank) {
            root_triplets.push_back(std::tuple<int, int, int>{
                co - slice_starts[leaf_process], root_process,
                root - slice_starts[root_process]});
          }

          // if (world_rank == 0) printf("%d %d %d %d %d %d\n", root_process,
          // leaf_process, slice_starts[root_process],
          // slice_starts[leaf_process], root, co);
          recvs[leaf_process].insert(root_process);
          sends[root_process].insert(leaf_process);
        }
      }

      std::vector<MPI_Request> reqs(0);
      std::vector<MatrixXf> yhats(world_size);

      for (int i : recvs[world_rank]) {
        reqs.push_back(MPI_Request());
        yhats[i] = MatrixXf::Zero(rows[i], cols[i]);
        MPI_Irecv(yhats[i].data(), rows[i] * cols[i], MPI_FLOAT, i, 0,
                  comm_global, &reqs[reqs.size() - 1]);
      }

      for (int i : sends[world_rank]) {
        reqs.push_back(MPI_Request());
        MPI_Isend(yhat.data(), rows[world_rank] * cols[world_rank], MPI_FLOAT,
                  i, 0, comm_global, &reqs[reqs.size() - 1]);
      }

      std::vector<MPI_Status> stats(reqs.size());
      MPI_Waitall(reqs.size(), reqs.data(), stats.data());

      MatrixXf y = MatrixXf::Zero(ro, co);

      for (auto&& p : root_triplets) {
        int leaf_index, root_process, root_index;
        std::tie(leaf_index, root_process, root_index) = p;
        y.middleRows(leaf_index, 1) =
            P(leaf_index + slice_starts[world_rank], 0) *
            yhats[root_process].middleRows(root_index, 1);
      }

      MatrixXf yhat_total = MatrixXf::Zero(num_total, co);

      curr_row = 0;
      for (int i = 0; i < world_size; i++) {
        if (i != world_rank) {
          yhats[i] = MatrixXf::Zero(rows[i], cols[i]);
        } else {
          yhats[i] = y;
        }
        MPI_Bcast(yhats[i].data(), rows[i] * cols[i], MPI_FLOAT, i,
                  comm_global);
        yhat_total.middleRows(curr_row, rows[i]) = yhats[i].eval();

        curr_row += rows[i];
      }

      SpMat S = S::build_sparse(S_compact, num_base, num_total, num_levels)
                    .middleRows(slice_starts[world_rank], rows[world_rank])
                    .eval();

      /*
if (world_rank == 0) {
    for (int i = 0; i < world_size; i++) {
        printf("Rank %d needs ", i);
        for (int k: recvs[i]) {
            printf("%d, ", k);
        }
        printf("\n");
    }
}
*/

      result = (S * yhat_total.topRows(num_base)).eval();

    } else if (method == "middle_out") {
      std::vector<int> slice_starts(world_size);
      std::vector<std::set<int>> recvs(world_size, std::set<int>());
      std::vector<std::set<int>> sends(world_size, std::set<int>());

      int curr_row = 0;

      for (int i = 0; i < world_size; i++) {
        slice_starts[i] = curr_row;
        curr_row += rows[i];
      }

      std::vector<std::tuple<int, int, int>> root_triplets(0);
      for (int i = 0; i < num_base; i++) {
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
          int root_process = 0, leaf_process = 0;
          for (int j = 0; j < world_size; j++) {
            if (slice_starts[j] + rows[j] > root) {
              root_process = j;
              break;
            }
          }

          for (int j = 0; j < world_size; j++) {
            if (slice_starts[j] + rows[j] > co) {
              leaf_process = j;
              break;
            }
          }

          if (leaf_process == world_rank) {
            root_triplets.push_back(std::tuple<int, int, int>{
                co - slice_starts[leaf_process], root_process,
                root - slice_starts[root_process]});
          }

          recvs[leaf_process].insert(root_process);
          sends[root_process].insert(leaf_process);
        }
      }

      std::vector<MPI_Request> reqs(0);
      std::vector<MatrixXf> yhats(world_size);

      for (int i : recvs[world_rank]) {
        reqs.push_back(MPI_Request());
        yhats[i] = MatrixXf::Zero(rows[i], cols[i]);
        MPI_Irecv(yhats[i].data(), rows[i] * cols[i], MPI_FLOAT, i, 0,
                  comm_global, &reqs[reqs.size() - 1]);
      }

      for (int i : sends[world_rank]) {
        reqs.push_back(MPI_Request());
        MPI_Isend(yhat.data(), rows[world_rank] * cols[world_rank], MPI_FLOAT,
                  i, 0, comm_global, &reqs[reqs.size() - 1]);
      }

      std::vector<MPI_Status> stats(reqs.size());
      MPI_Waitall(reqs.size(), reqs.data(), stats.data());

      MatrixXf y = MatrixXf::Zero(ro, co);

      for (auto&& p : root_triplets) {
        int leaf_index, root_process, root_index;
        std::tie(leaf_index, root_process, root_index) = p;
        y.middleRows(leaf_index, 1) =
            P(leaf_index + slice_starts[world_rank], 0) *
            yhats[root_process].middleRows(root_index, 1);
      }

      MatrixXf yhat_total = MatrixXf::Zero(num_total, co);

      curr_row = 0;
      for (int i = 0; i < world_size; i++) {
        if (i != world_rank) {
          yhats[i] = MatrixXf::Zero(rows[i], cols[i]);
        } else {
          yhats[i] = y;
        }
        MPI_Bcast(yhats[i].data(), rows[i] * cols[i], MPI_FLOAT, i,
                  comm_global);
        yhat_total.middleRows(curr_row, rows[i]) = yhats[i].eval();

        curr_row += rows[i];
      }

      SpMat S = S::build_sparse(S_compact, num_base, num_total, num_levels)
                    .middleRows(slice_starts[world_rank], rows[world_rank])
                    .eval();

      result = (S * yhat_total.topRows(num_base)).eval();
    }
    /* else if (method == "OLS") {
    result = yhat;
}
else if (method == "WLS") {
    result = yhat;
} */
    else {
      throw std::invalid_argument(
          "invalid reconciliation method. Available options are: bottom_up, "
          "top_down, middle_out");  //, OLS, WLS");
    }

    return result;
  }

  MatrixXf reconcile_dp_matrix(const std::string method,
                               const MatrixXi S_compact, const MatrixXf P,
                               const MatrixXf yhat, int level, float w,
                               int num_base, int num_total, int num_levels) {
    int world_size;
    MPI_Comm_size(comm_global, &world_size);
    int world_rank;
    MPI_Comm_rank(comm_global, &world_rank);

    int ro = yhat.rows();
    int co = yhat.cols();

    std::vector<int> rows(world_size);
    std::vector<int> cols(world_size);

    std::vector<MPI_Request> reqs(world_size);
    std::vector<MPI_Status> stats(world_size);

    MPI_Allgather(&ro, 1, MPI_INT, rows.data(), 1, MPI_INT, comm_global);
    MPI_Allgather(&co, 1, MPI_INT, cols.data(), 1, MPI_INT, comm_global);

    if (world_rank == 0) {
      int n_cols = cols[0];
      for (int i = 1; i < world_size; i++) {
        if (cols[i] != n_cols) {
          char buffer[200];
          sprintf(buffer, "Error: cols[%d] != cols[0]\n", i);
          throw std::invalid_argument(buffer);
        }
      }
    }

    MatrixXf yhat_total =
        MatrixXf::Zero(std::accumulate(rows.begin(), rows.end(), 0), cols[0]);
    std::vector<MatrixXf> yhats(world_size);

    int slice_start = 0, slice_length = 0;
    int curr_row = 0;

    for (int i = 0; i < world_size; i++) {
      if (i != world_rank) {
        yhats[i] = MatrixXf::Zero(rows[i], cols[i]);
      } else {
        yhats[i] = yhat;
      }
      MPI_Bcast(yhats[i].data(), rows[i] * cols[i], MPI_FLOAT, i, comm_global);
      yhat_total.middleRows(curr_row, rows[i]) = yhats[i].eval();

      if (i == world_rank) {
        slice_start = curr_row;
        slice_length = rows[i];
      }

      curr_row += rows[i];
    }

    // printf("rank %d: %d %d\n", world_rank, slice_start, slice_length);

    return dp_reconcile_optimized(method, S_compact, P, yhat_total, level, w,
                                  num_base, num_total, num_levels, slice_start,
                                  slice_length);

    /*
MPI_Barrier(comm_global);

MatrixXf reconciliation_matrix =
    construct_dp_reconciliation_matrix(method,
        S_compact, P, level, w, num_base, num_total, num_levels,
slice_start, slice_length);


return reconciliation_matrix * yhat_total;
*/

    /*
 if (world_rank == world_size - 1) {
     std::stringstream ss;
     ss << reconciliation_matrix; //(seqN(0, 5), all);
     printf("y_return: %s\n", ss.str().c_str());
 }
 */
  }

  MatrixXf reconcile_gather(const std::string method, const MatrixXi S_compact,
                            const MatrixXf P, const MatrixXf yhat, int level,
                            float w, int num_base, int num_total,
                            int num_levels) {
    int world_size;
    MPI_Comm_size(comm_global, &world_size);
    int world_rank;
    MPI_Comm_rank(comm_global, &world_rank);

    int ro = yhat.rows();
    int co = yhat.cols();

    std::vector<int> rows(world_size);
    std::vector<int> cols(world_size);

    std::vector<MPI_Request> reqs(world_size);
    std::vector<MPI_Status> stats(world_size);

    MPI_Gather(&ro, 1, MPI_INT, rows.data(), 1, MPI_INT, 0, comm_global);
    MPI_Gather(&co, 1, MPI_INT, cols.data(), 1, MPI_INT, 0, comm_global);

    MatrixXf yhat_total;
    std::vector<MatrixXf> yhats(world_size);

    if (world_rank == 0) {
      int n_cols = cols[0];
      for (int i = 1; i < world_size; i++) {
        if (cols[i] != n_cols) {
          char buffer[200];
          sprintf(buffer, "Error: cols[%d] != cols[0]\n", i);
          throw std::invalid_argument(buffer);
        }
      }

      yhat_total =
          MatrixXf::Zero(std::accumulate(rows.begin(), rows.end(), 0), cols[0]);

      for (int i = 1; i < world_size; i++) {
        yhats[i] = MatrixXf::Zero(rows[i], cols[i]);
        MPI_Irecv(yhats[i].data(), rows[i] * cols[i], MPI_FLOAT, i, 0,
                  comm_global, &reqs[i]);
      }

      MPI_Waitall(world_size, reqs.data(), stats.data());

      int curr_row = rows[0];

      yhat_total.topRows(rows[0]) = yhat.eval();

      for (int i = 1; i < world_size; i++) {
        yhat_total.middleRows(curr_row, rows[i]) = yhats[i].eval();
        curr_row += rows[i];
      }

    } else {
      MPI_Isend(yhat.data(), ro * co, MPI_FLOAT, 0, 0, comm_global, &reqs[0]);
      MPI_Wait(&reqs[0], &stats[0]);
    }

    MatrixXf y_return;

    if (world_rank == 0) {
      omp_set_num_threads(24);

      MatrixXf y_reconciled = reconcile(method, S_compact, P, yhat_total, level,
                                        w, num_base, num_total, num_levels);

      y_return = y_reconciled.topRows(rows[0]).eval();

      // std::stringstream ss;
      // ss << y_reconciled(seqN(0, 5), all);
      // printf("y_return: %s\n", ss.str().c_str());

      int curr_row = rows[0];
      for (int i = 1; i < world_size; i++) {
        yhats[i] = y_reconciled.middleRows(curr_row, rows[i]).eval();
        MPI_Isend(yhats[i].data(), rows[i] * cols[i], MPI_FLOAT, i, 0,
                  comm_global, &reqs[i]);
        curr_row += rows[i];
      }

      MPI_Waitall(world_size, reqs.data(), stats.data());
      MPI_Barrier(comm_global);
      return y_return;
    } else {
      y_return = MatrixXf::Zero(ro, co);
      MPI_Irecv(y_return.data(), ro * co, MPI_FLOAT, 0, 0, comm_global,
                &reqs[0]);
      MPI_Wait(&reqs[0], &stats[0]);
      MPI_Barrier(comm_global);
      return y_return;
    }
  }

  void test(const MatrixXd& xs) {
    int world_size;
    MPI_Comm_size(comm_global, &world_size);
    int world_rank;
    MPI_Comm_rank(comm_global, &world_rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME] = "localhost";
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    MPI_Status status;
    if (world_rank == 0) {
      MPI_Send((void*)xs.data(), 9, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    } else if (world_rank == 1) {
      MPI_Recv((void*)xs.data(), 9, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
#ifdef _OPENMP
      printf("OpenMP enabled.\n");
#endif
      printf("%s, MPI rank %d out of %d, %f\n", processor_name, world_rank,
             world_size, xs.determinant());
    }

    if (world_rank == 0) {
      int thread_level;
      MPI_Query_thread(&thread_level);
      switch (thread_level) {
        case MPI_THREAD_SINGLE:
          printf("Detected thread level MPI_THREAD_SINGLE\n");
          fflush(stdout);
          break;
        case MPI_THREAD_FUNNELED:
          printf("Detected thread level MPI_THREAD_FUNNELED\n");
          fflush(stdout);
          break;
        case MPI_THREAD_SERIALIZED:
          printf("Detected thread level MPI_THREAD_SERIALIZED\n");
          fflush(stdout);
          break;
        case MPI_THREAD_MULTIPLE:
          printf("Detected thread level MPI_THREAD_MULTIPLE\n");
          fflush(stdout);
          break;
      }

      int nthreads, tid;

#pragma omp parallel private(nthreads, tid)
      {
        /* Obtain thread number */
        tid = omp_get_thread_num();
        printf("Hello World from thread = %d\n", tid);

        /* Only master thread does this */
        if (tid == 0) {
          nthreads = omp_get_num_threads();
          printf("Number of threads = %d\n", nthreads);
        }
      }
    }
  }

 private:
  MPI_Comm comm_global;
};

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

  py::class_<MPI_Utils>(m, "MPI_Utils")
      .def(py::init<>())
      .def("reconcile_gather", &MPI_Utils::reconcile_gather, "reconcile_gather")
      .def("reconcile_dp_matrix", &MPI_Utils::reconcile_dp_matrix,
           "reconcile_dp_matrix")
      .def("reconcile_dp_optimized", &MPI_Utils::reconcile_dp_optimized,
           "reconcile_dp_matrix");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}