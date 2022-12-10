#include "Distributed.h"

using namespace lhts;

MatrixXd dp_reconcile(const std::string method,
                                const MatrixXi S_compact, const MatrixXd P,
                                const MatrixXd yhat, int level, double w,
                                int num_leaves, int num_nodes, int num_levels,
                                int slice_start, int slice_length) {
  SpMat S = S::build_sparse(S_compact, num_leaves, num_nodes, num_levels)
                .middleRows(slice_start, slice_length)
                .eval();
  SpMat G, res;
  MatrixXd result, y;
  y = yhat;

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


MatrixXd Distributed::reconcile_dp_optimized(const std::string method, const MatrixXi S_compact,
                      int num_leaves, int num_nodes, int num_levels, const MatrixXd yhat,
                       const MatrixXd P, int level, double w) {
  omp_set_num_threads(8);
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

  MatrixXd result;

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

    MatrixXd yhat_total =
        MatrixXd::Zero(std::accumulate(rows.begin(), rows.end(), 0), cols[0]);
    std::vector<MatrixXd> yhats(world_size);

    curr_row = 0;
    for (int i = 0; i < world_size; i++) {
      MPI_Comm leaf_comm;

      int color =
          (i == world_rank) | (slice_start + slice_length >= num_leaves);
      MPI_Comm_split(comm_global, color,
                     (i == world_rank) ? 0 : world_rank + world_size,
                     &leaf_comm);

      if (color == 1) {
        if (i != world_rank) {
          yhats[i] = MatrixXd::Zero(rows[i], cols[i]);
        } else {
          yhats[i] = yhat;
        }

        MPI_Bcast(yhats[i].data(), rows[i] * cols[i], MPI_DOUBLE, 0, leaf_comm);
        yhat_total.middleRows(curr_row, rows[i]) = yhats[i].eval();

        MPI_Comm_free(&leaf_comm);
      }

      curr_row += rows[i];
    }

    if (slice_start + slice_length >= num_leaves) {
      result = dp_reconcile(method, S_compact, P, yhat_total, level,
                                      w, num_leaves, num_nodes, num_levels,
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

    std::vector<std::tuple<int, int, int>> root_triplets(0);

    for (int i = 0; i < num_leaves; i++) {
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
    std::vector<MatrixXd> yhats(world_size);

    for (int i : recvs[world_rank]) {
      reqs.push_back(MPI_Request());
      yhats[i] = MatrixXd::Zero(rows[i], cols[i]);
      MPI_Irecv(yhats[i].data(), rows[i] * cols[i], MPI_DOUBLE, i, 0,
                comm_global, &reqs[reqs.size() - 1]);
    }

    for (int i : sends[world_rank]) {
      reqs.push_back(MPI_Request());
      MPI_Isend(yhat.data(), rows[world_rank] * cols[world_rank], MPI_DOUBLE, i,
                0, comm_global, &reqs[reqs.size() - 1]);
    }

    std::vector<MPI_Status> stats(reqs.size());
    MPI_Waitall(reqs.size(), reqs.data(), stats.data());

    MatrixXd y = MatrixXd::Zero(ro, co);

    for (auto&& p : root_triplets) {
      int leaf_index, root_process, root_index;
      std::tie(leaf_index, root_process, root_index) = p;
      y.middleRows(leaf_index, 1) =
          P(leaf_index + slice_starts[world_rank], 0) *
          yhats[root_process].middleRows(root_index, 1);
    }

    MatrixXd yhat_total = MatrixXd::Zero(num_nodes, co);

    curr_row = 0;
    for (int i = 0; i < world_size; i++) {
      if (i != world_rank) {
        yhats[i] = MatrixXd::Zero(rows[i], cols[i]);
      } else {
        yhats[i] = y;
      }
      MPI_Bcast(yhats[i].data(), rows[i] * cols[i], MPI_DOUBLE, i, comm_global);
      yhat_total.middleRows(curr_row, rows[i]) = yhats[i].eval();

      curr_row += rows[i];
    }

    SpMat S = S::build_sparse(S_compact, num_leaves, num_nodes, num_levels)
                  .middleRows(slice_starts[world_rank], rows[world_rank])
                  .eval();

    result = (S * yhat_total.topRows(num_leaves)).eval();

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
    for (int i = 0; i < num_leaves; i++) {
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
    std::vector<MatrixXd> yhats(world_size);

    for (int i : recvs[world_rank]) {
      reqs.push_back(MPI_Request());
      yhats[i] = MatrixXd::Zero(rows[i], cols[i]);
      MPI_Irecv(yhats[i].data(), rows[i] * cols[i], MPI_DOUBLE, i, 0,
                comm_global, &reqs[reqs.size() - 1]);
    }

    for (int i : sends[world_rank]) {
      reqs.push_back(MPI_Request());
      MPI_Isend(yhat.data(), rows[world_rank] * cols[world_rank], MPI_DOUBLE, i,
                0, comm_global, &reqs[reqs.size() - 1]);
    }

    std::vector<MPI_Status> stats(reqs.size());
    MPI_Waitall(reqs.size(), reqs.data(), stats.data());

    MatrixXd y = MatrixXd::Zero(ro, co);

    for (auto&& p : root_triplets) {
      int leaf_index, root_process, root_index;
      std::tie(leaf_index, root_process, root_index) = p;
      y.middleRows(leaf_index, 1) =
          P(leaf_index + slice_starts[world_rank], 0) *
          yhats[root_process].middleRows(root_index, 1);
    }

    MatrixXd yhat_total = MatrixXd::Zero(num_nodes, co);

    curr_row = 0;
    for (int i = 0; i < world_size; i++) {
      if (i != world_rank) {
        yhats[i] = MatrixXd::Zero(rows[i], cols[i]);
      } else {
        yhats[i] = y;
      }
      MPI_Bcast(yhats[i].data(), rows[i] * cols[i], MPI_DOUBLE, i, comm_global);
      yhat_total.middleRows(curr_row, rows[i]) = yhats[i].eval();

      curr_row += rows[i];
    }

    SpMat S = S::build_sparse(S_compact, num_leaves, num_nodes, num_levels)
                  .middleRows(slice_starts[world_rank], rows[world_rank])
                  .eval();

    result = (S * yhat_total.topRows(num_leaves)).eval();
  } else {
    throw std::invalid_argument(
        "invalid reconciliation method. Available options are: bottom_up, "
        "top_down, middle_out");
  }

  return result;
}

MatrixXd Distributed::reconcile_dp_matrix(const std::string method, const MatrixXi S_compact,
                      int num_leaves, int num_nodes, int num_levels, const MatrixXd yhat,
                       const MatrixXd P, int level, double w) {
  omp_set_num_threads(8);
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

  MatrixXd yhat_total =
      MatrixXd::Zero(std::accumulate(rows.begin(), rows.end(), 0), cols[0]);
  std::vector<MatrixXd> yhats(world_size);

  int slice_start = 0, slice_length = 0;
  int curr_row = 0;

  for (int i = 0; i < world_size; i++) {
    if (i != world_rank) {
      yhats[i] = MatrixXd::Zero(rows[i], cols[i]);
    } else {
      yhats[i] = yhat;
    }
    MPI_Bcast(yhats[i].data(), rows[i] * cols[i], MPI_DOUBLE, i, comm_global);
    yhat_total.middleRows(curr_row, rows[i]) = yhats[i].eval();

    if (i == world_rank) {
      slice_start = curr_row;
      slice_length = rows[i];
    }

    curr_row += rows[i];
  }

  return dp_reconcile(method, S_compact, P, yhat_total, level, w,
                                num_leaves, num_nodes, num_levels, slice_start,
                                slice_length);

}

MatrixXd Distributed::reconcile_gather(const std::string method, const MatrixXi S_compact,
                      int num_leaves, int num_nodes, int num_levels, const MatrixXd yhat,
                       const MatrixXd P, int level, double w) {
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

  MatrixXd yhat_total;
  std::vector<MatrixXd> yhats(world_size);

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
        MatrixXd::Zero(std::accumulate(rows.begin(), rows.end(), 0), cols[0]);

    for (int i = 1; i < world_size; i++) {
      yhats[i] = MatrixXd::Zero(rows[i], cols[i]);
      MPI_Irecv(yhats[i].data(), rows[i] * cols[i], MPI_DOUBLE, i, 0,
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
    MPI_Isend(yhat.data(), ro * co, MPI_DOUBLE, 0, 0, comm_global, &reqs[0]);
    MPI_Wait(&reqs[0], &stats[0]);
  }

  MatrixXd y_return;

  if (world_rank == 0) {
    omp_set_num_threads(8);

    MatrixXd y_reconciled =
        reconcile::sparse_matrix(method, S_compact,
                       num_leaves, num_nodes, num_levels, yhat_total,
                        P, level, w);

    y_return = y_reconciled.topRows(rows[0]).eval();

    int curr_row = rows[0];
    for (int i = 1; i < world_size; i++) {
      yhats[i] = y_reconciled.middleRows(curr_row, rows[i]).eval();
      MPI_Isend(yhats[i].data(), rows[i] * cols[i], MPI_DOUBLE, i, 0,
                comm_global, &reqs[i]);
      curr_row += rows[i];
    }

    MPI_Waitall(world_size, reqs.data(), stats.data());
    MPI_Barrier(comm_global);
    return y_return;
  } else {
    y_return = MatrixXd::Zero(ro, co);
    MPI_Irecv(y_return.data(), ro * co, MPI_DOUBLE, 0, 0, comm_global, &reqs[0]);
    MPI_Wait(&reqs[0], &stats[0]);
    MPI_Barrier(comm_global);
    return y_return;
  }
}

void Distributed::test(const MatrixXd& xs) {
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
