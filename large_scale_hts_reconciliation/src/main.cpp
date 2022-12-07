// Referencing https://github.com/latug0/pybind_mpi/

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <vector>
#include <set>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>
#include <stdio.h>
#include <Eigen/LU>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


Eigen::MatrixXi construct_S(const Eigen::MatrixXi S_compact, int num_base, int num_total, int num_levels) {
    Eigen::MatrixXi S = Eigen::MatrixXi::Zero(num_total, num_base);
    
    assert(S_compact.rows() == num_total);
    assert(S_compact.cols() == num_levels);
    assert(num_levels > 1);

    #pragma omp parallel for 
    for (int i = 0; i < num_base; i++) {
        int co = S_compact(i, 0);
        if (co >= num_base) {
            throw std::invalid_argument("Make sure that the frist num_base rows of S_compact contain only leaf-level nodes.");
        }
        S(co, co) = 1;
        for (int j = 1; j < num_levels; j++) {
            int ro = S_compact(i, j);
            if (ro == -1) {
                if (i < num_base) {
                    throw std::invalid_argument("Make sure that the frist num_base rows of S_compact contain only leaf-level nodes.");
                }
                break;
            } else {
                if (co >= num_base) {
                    throw std::invalid_argument("Make sure that the all leaf-level nodes have index < num_base.");
                }
                S(ro, co) = 1;
            }
        }
    }

    return S;
}

Eigen::MatrixXf construct_G_OLS(const Eigen::MatrixXi S) {
    Eigen::MatrixXf Sp = S.cast<float>().eval();
    Eigen::MatrixXf St = Sp.transpose().eval();
    Eigen::MatrixXf M = St * Sp;
    Eigen::FullPivLU<Eigen::MatrixXf> lu(M);
    return lu.matrixLU() * St;
}

Eigen::MatrixXf construct_G_WLS(const Eigen::MatrixXi S, float w) {
    Eigen::MatrixXf W = Eigen::MatrixXf::Zero(S.rows(), S.cols());
    #pragma omp parallel for 
    for (int i = 0; i < S.rows(); i++) {
        W(i, i) = w;
    }
    Eigen::MatrixXf Sp = S.cast<float>();
    Eigen::MatrixXf St = Sp.transpose();
    Eigen::MatrixXf M = St * W * Sp;
    Eigen::FullPivLU<Eigen::MatrixXf> lu(M);
    return lu.matrixLU() * St * W;
}


Eigen::MatrixXf distribute_forecast_top_down(const Eigen::MatrixXi S_compact, 
                const Eigen::MatrixXf P, const Eigen::MatrixXf yhat, 
                int num_base, int num_total, int num_levels) {
    
    Eigen::MatrixXf y = Eigen::MatrixXf::Zero(num_base, yhat.cols());
    
    assert(S_compact.rows() == num_total);
    assert(S_compact.cols() == num_levels);
    assert(num_levels > 1);

    #pragma omp parallel for 
    for (int i = 0; i < num_total; i++) {
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
            y.middleRows(co, 1) = P(co, 0) * yhat.middleRows(root, 1);
        }
    }

    return y;
}


Eigen::MatrixXf distribute_forecast_middle_out(const Eigen::MatrixXi S_compact, 
                const Eigen::MatrixXf P, const Eigen::MatrixXf yhat, 
                int level, int num_base, int num_total, int num_levels) {
    
    Eigen::MatrixXf y = Eigen::MatrixXf::Zero(num_base, yhat.cols());
    
    assert(S_compact.rows() == num_total);
    assert(S_compact.cols() == num_levels);
    assert(num_levels > 1);

    #pragma omp parallel for 
    for (int i = 0; i < num_total; i++) {
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
            y.middleRows(co, 1) = P(co, 0) * yhat.middleRows(root, 1);
        }
    }

    return y;
}


Eigen::MatrixXf construct_G_top_down(const Eigen::MatrixXi S_compact, 
                const Eigen::MatrixXf P, 
                int num_base, int num_total, int num_levels) {
    Eigen::MatrixXf G = Eigen::MatrixXf::Zero(num_base, num_total);
    
    assert(S_compact.rows() == num_total);
    assert(S_compact.cols() == num_levels);
    assert(num_levels > 1);

    #pragma omp parallel for 
    for (int i = 0; i < num_total; i++) {
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


Eigen::MatrixXf construct_G_middle_out(const Eigen::MatrixXi S_compact, 
                const Eigen::MatrixXf P, int level, 
                int num_base, int num_total, int num_levels) {
    Eigen::MatrixXf G = Eigen::MatrixXf::Zero(num_base, num_total);
    
    assert(S_compact.rows() == num_total);
    assert(S_compact.cols() == num_levels);
    assert(num_levels > 1);

    #pragma omp parallel for 
    for (int i = 0; i < num_total; i++) {
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


Eigen::MatrixXf construct_G_bottom_up(const Eigen::MatrixXi S_compact, int num_base, int num_total, int num_levels) {
    Eigen::MatrixXf G = Eigen::MatrixXf::Zero(num_base, num_total);
    
    assert(S_compact.rows() == num_total);
    assert(S_compact.cols() == num_levels);
    assert(num_levels > 1);

    #pragma omp parallel for 
    for (int i = 0; i < num_total; i++) {
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

Eigen::MatrixXf dp_reconcile_optimized(const std::string method,
                          const Eigen::MatrixXi S_compact,
                          const Eigen::MatrixXf P,
                          const Eigen::MatrixXf yhat,
                          int level, float w,
                          int num_base, int num_total, int num_levels,
                          int slice_start, int slice_length) {
    Eigen::MatrixXi S = construct_S(S_compact, num_base, num_total, num_levels).middleRows(slice_start, slice_length).eval();
    Eigen::MatrixXf G, res, y;
    y = yhat;
    
    if (method == "bottom_up") {
        res = S.cast<float>();
        y = yhat.topRows(num_base).eval();    
    }
    else if (method == "top_down") {
        res = S.cast<float>();
        y = distribute_forecast_top_down(S_compact, P, yhat, num_base, num_total, num_levels);
    }
    else if (method == "middle_out") {
        res = S.cast<float>();
        y = distribute_forecast_middle_out(S_compact, P, yhat, level, num_base, num_total, num_levels);
    }
    else if (method == "OLS") {
        G = construct_G_OLS(S);
        res = S.cast<float>() * G;
    }
    else if (method == "WLS") {
        G = construct_G_WLS(S, w);
        res = S.cast<float>() * G;
    }
    else {
        throw std::invalid_argument("invalid reconciliation method. Available options are: bottom_up, top_down, middle_out, OLS, WLS");
    }

    res = res * y;

    return res;
}


Eigen::MatrixXf construct_dp_reconciliation_matrix(const std::string method,
                          const Eigen::MatrixXi S_compact,
                          const Eigen::MatrixXf P,
                          int level, float w,
                          int num_base, int num_total, int num_levels,
                          int slice_start, int slice_length) {
    Eigen::MatrixXi S = construct_S(S_compact, num_base, num_total, num_levels);
    
    Eigen::MatrixXf G;
    
    if (method == "bottom_up") {
        G = construct_G_bottom_up(S_compact, num_base, num_total, num_levels);
    }
    else if (method == "top_down") {
        G = construct_G_top_down(S_compact, P, num_base, num_total, num_levels);
    }
    else if (method == "middle_out") {
        G = construct_G_middle_out(S_compact, P, level, num_base, num_total, num_levels);
    }
    else if (method == "OLS") {
        G = construct_G_OLS(S);
    }
    else if (method == "WLS") {
        G = construct_G_WLS(S, w);
    }
    else {
        throw std::invalid_argument("invalid reconciliation method. Available options are: bottom_up, top_down, middle_out, OLS, WLS");
    }

    Eigen::MatrixXi S_slice = S.middleRows(slice_start, slice_length).eval();
    Eigen::MatrixXf res = (S_slice.cast<float>() * G).eval();

    return res;
}


Eigen::MatrixXf reconcile_matrix(const std::string method,
                          const Eigen::MatrixXi S_compact,
                          const Eigen::MatrixXf P,
                          const Eigen::MatrixXf yhat,
                          int level, float w,
                          int num_base, int num_total, int num_levels) {

    Eigen::MatrixXi S = construct_S(S_compact, num_base, num_total, num_levels);

    Eigen::MatrixXf G, res, y;
    y = yhat;
    
    if (method == "bottom_up") {
        G = construct_G_bottom_up(S_compact, num_base, num_total, num_levels);
        res = S.cast<float>() * G;
    }
    else if (method == "top_down") {
        G = construct_G_top_down(S_compact, P, num_base, num_total, num_levels);
        res = S.cast<float>() * G;
    }
    else if (method == "middle_out") {
        G = construct_G_middle_out(S_compact, P, level, num_base, num_total, num_levels);
        res = S.cast<float>() * G;
    }
    else if (method == "OLS") {
        G = construct_G_OLS(S);
        res = S.cast<float>() * G;
    }
    else if (method == "WLS") {
        G = construct_G_WLS(S, w);
        res = S.cast<float>() * G;
    }
    else {
        throw std::invalid_argument("invalid reconciliation method. Available options are: bottom_up, top_down, middle_out, OLS, WLS");
    }

    res = res * y;

    return res;
}


Eigen::MatrixXf reconcile(const std::string method,
                          const Eigen::MatrixXi S_compact,
                          const Eigen::MatrixXf P,
                          const Eigen::MatrixXf yhat,
                          int level, float w,
                          int num_base, int num_total, int num_levels) {

    Eigen::MatrixXi S = construct_S(S_compact, num_base, num_total, num_levels);
    
    // std::stringstream ss;
    // ss << S.rows() << " " << S.cols() << " " << S(Eigen::seqN(0, 10), Eigen::seqN(0, 10));
    // printf("S: %s\n", ss.str().c_str());

    Eigen::MatrixXf G, res, y;
    y = yhat;
    
    if (method == "bottom_up") {
        res = S.cast<float>();
        y = yhat.topRows(num_base).eval();    
    }
    else if (method == "top_down") {
        res = S.cast<float>();
        y = distribute_forecast_top_down(S_compact, P, yhat, num_base, num_total, num_levels);
    }
    else if (method == "middle_out") {
        res = S.cast<float>();
        y = distribute_forecast_middle_out(S_compact, P, yhat, level, num_base, num_total, num_levels);
    }
    else if (method == "OLS") {
        G = construct_G_OLS(S);
        res = S.cast<float>() * G;
    }
    else if (method == "WLS") {
        G = construct_G_WLS(S, w);
        res = S.cast<float>() * G;
    }
    else {
        throw std::invalid_argument("invalid reconciliation method. Available options are: bottom_up, top_down, middle_out, OLS, WLS");
    }
    
    res = res * y;

    return res;
}

float rmse(const Eigen::MatrixXf res, const Eigen::MatrixXf gt) {
    float sum = 0;
    for (int i = 0; i < res.rows(); i ++) {
        for (int j = 0; j < res.cols(); j ++) {
            sum += pow(abs(res(i, j) - gt(i, j)), 2);
        }
    }
    float rmse = sqrt(sum / (res.rows() * res.cols()));
    return rmse;
}

float mae(const Eigen::MatrixXf res, const Eigen::MatrixXf gt) {
    float sum = 0;
    for (int i = 0; i < res.rows(); i ++) {
        for (int j = 0; j < res.cols(); j ++) {
            sum += abs(res(i, j) - gt(i, j));
        }
    }
    float mae = sum / (res.rows() * res.cols());
    return mae;
}

float smape(const Eigen::MatrixXf res, const Eigen::MatrixXf gt) {
    float sum = 0;
    for (int i = 0; i < res.rows(); i ++) {
        for (int j = 0; j < res.cols(); j ++) {
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

class MPI_Utils
{
public:
  MPI_Utils() : comm_global(MPI_COMM_WORLD) {
    Eigen::initParallel();
  }
  
  ~MPI_Utils() {}

  Eigen::MatrixXf reconcile_dp_optimized(const std::string method,
                                const Eigen::MatrixXi S_compact,
                                const Eigen::MatrixXf P,
                                const Eigen::MatrixXf yhat,
                                int level, float w,
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

    Eigen::MatrixXf result;

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
        
        Eigen::MatrixXf yhat_total = Eigen::MatrixXf::Zero(std::accumulate(rows.begin(), rows.end(), 0), 
                                            cols[0]);
        std::vector<Eigen::MatrixXf> yhats(world_size);

        curr_row = 0;
        for (int i = 0; i < world_size; i++) {
            MPI_Comm leaf_comm;

            int color = (i == world_rank) | (slice_start + slice_length >= num_base);
            MPI_Comm_split(comm_global, color, (i == world_rank) ? 0 : world_rank + world_size, &leaf_comm);
            
            if (color == 1) {
                if (i != world_rank) {
                    yhats[i] = Eigen::MatrixXf::Zero(rows[i], cols[i]);
                } else {
                    yhats[i] = yhat;
                }
                // printf("rank %d @ %d, %d-%d\n", world_rank, i, slice_start, slice_length);

                MPI_Bcast(yhats[i].data(), rows[i] * cols[i], MPI_FLOAT, 0, leaf_comm);
                yhat_total.middleRows(curr_row, rows[i]) = yhats[i].eval();

                MPI_Comm_free(&leaf_comm);
            }

            curr_row += rows[i];
        }

        if (slice_start + slice_length >= num_base) {
            result = dp_reconcile_optimized(method, S_compact, P, yhat_total, level, w, 
                        num_base, num_total, num_levels, slice_start, slice_length);
        } else {
            result = yhat.eval();
        }
        
    }
    else if (method == "top_down") {

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
                
                // if (world_rank == 0) printf("%d %d %d %d %d %d\n", root_process, leaf_process, slice_starts[root_process], slice_starts[leaf_process], root, co);
                recvs[leaf_process].insert(root_process);
                sends[root_process].insert(leaf_process);
                
            }
        }

        //printf("insert\n");

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

        result = yhat;
    }
    else if (method == "middle_out") {
        result = yhat;
    }
    else if (method == "OLS") {
        result = yhat;
    }
    else if (method == "WLS") {
        result = yhat;
    }
    else {
        throw std::invalid_argument("invalid reconciliation method. Available options are: bottom_up, top_down, middle_out, OLS, WLS");
    }

    return result;

  }


  Eigen::MatrixXf reconcile_dp_matrix(const std::string method,
                                const Eigen::MatrixXi S_compact,
                                const Eigen::MatrixXf P,
                                const Eigen::MatrixXf yhat,
                                int level, float w,
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

    Eigen::MatrixXf yhat_total = Eigen::MatrixXf::Zero(std::accumulate(rows.begin(), rows.end(), 0), 
                                           cols[0]);
    std::vector<Eigen::MatrixXf> yhats(world_size);

    int slice_start = 0, slice_length = 0;
    int curr_row = 0;
    
    for (int i = 0; i < world_size; i++) {

        if (i != world_rank) {
            yhats[i] = Eigen::MatrixXf::Zero(rows[i], cols[i]);
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

    //printf("rank %d: %d %d\n", world_rank, slice_start, slice_length);

    return dp_reconcile_optimized(method, S_compact, P, yhat_total, level, w, 
                        num_base, num_total, num_levels, slice_start, slice_length);
    
    /*
    MPI_Barrier(comm_global);

    Eigen::MatrixXf reconciliation_matrix = 
        construct_dp_reconciliation_matrix(method, 
            S_compact, P, level, w, num_base, num_total, num_levels, slice_start, slice_length);


    return reconciliation_matrix * yhat_total;
    */

   /*
    if (world_rank == world_size - 1) {
        std::stringstream ss;
        ss << reconciliation_matrix; //(Eigen::seqN(0, 5), Eigen::all);
        printf("y_return: %s\n", ss.str().c_str());
    }
    */

  }

  Eigen::MatrixXf reconcile_gather(const std::string method,
                                    const Eigen::MatrixXi S_compact,
                                    const Eigen::MatrixXf P,
                                    const Eigen::MatrixXf yhat,
                                    int level, float w,
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

    MPI_Gather(&ro, 1, MPI_INT, rows.data(), 1, MPI_INT, 0, comm_global);
    MPI_Gather(&co, 1, MPI_INT, cols.data(), 1, MPI_INT, 0, comm_global);

    Eigen::MatrixXf yhat_total;
    std::vector<Eigen::MatrixXf> yhats(world_size);

    if (world_rank == 0) {
        int n_cols = cols[0];
        for (int i = 1; i < world_size; i++) {
            if (cols[i] != n_cols) {
                char buffer[200];
                sprintf(buffer, "Error: cols[%d] != cols[0]\n", i);
                throw std::invalid_argument(buffer);
            }
        }
        

        yhat_total = Eigen::MatrixXf::Zero(std::accumulate(rows.begin(), rows.end(), 0), 
                                           cols[0]);

        for (int i = 1; i < world_size; i++) {
            yhats[i] = Eigen::MatrixXf::Zero(rows[i], cols[i]);
            MPI_Irecv(yhats[i].data(), rows[i] * cols[i], MPI_FLOAT, i, 0, comm_global, &reqs[i]);
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

    Eigen::MatrixXf y_return;

    if (world_rank == 0) {
        omp_set_num_threads(24);
        
        Eigen::MatrixXf y_reconciled = reconcile(method, S_compact, P, yhat_total, level, w, num_base, num_total, num_levels);
    
        y_return = y_reconciled.topRows(rows[0]).eval();

        // std::stringstream ss;
        // ss << y_reconciled(Eigen::seqN(0, 5), Eigen::all);
        // printf("y_return: %s\n", ss.str().c_str());

        int curr_row = rows[0];
        for (int i = 1; i < world_size; i++) {
            yhats[i] = y_reconciled.middleRows(curr_row, rows[i]).eval();
            MPI_Isend(yhats[i].data(), rows[i] * cols[i], MPI_FLOAT, i, 0, comm_global, &reqs[i]);
            curr_row += rows[i];
        }

        MPI_Waitall(world_size, reqs.data(), stats.data());
        MPI_Barrier(comm_global);
        return y_return;
    } else {

        y_return = Eigen::MatrixXf::Zero(ro, co);
        MPI_Irecv(y_return.data(), ro * co, MPI_FLOAT, 0, 0, comm_global, &reqs[0]);
        MPI_Wait(&reqs[0], &stats[0]);
        MPI_Barrier(comm_global);
        return y_return;
    }

  }
  
  void test(const Eigen::MatrixXd &xs) {
    int world_size;
    MPI_Comm_size(comm_global, &world_size);
    int world_rank;
    MPI_Comm_rank(comm_global, &world_rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME] = "localhost";
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    
    MPI_Status status;
    if (world_rank == 0)
    {
        MPI_Send((void*)xs.data(), 9, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 1)
    {
        MPI_Recv((void*)xs.data(), 9, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        #ifdef _OPENMP
        printf("OpenMP enabled.\n");
        #endif
        printf("%s, MPI rank %d out of %d, %f\n",
            processor_name,
            world_rank,
            world_size,
            xs.determinant());
    }

    if (world_rank == 0) {
        int thread_level;
        MPI_Query_thread( &thread_level );
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
	if (tid == 0 ) 
	  {
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
        Pybind11 example plugin
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
    m.def("construct_S", &construct_S);
    m.def("construct_G_bottom_up", &construct_G_bottom_up);
    m.def("construct_G_top_down", &construct_G_top_down);
    m.def("construct_G_middle_out", &construct_G_middle_out);

    py::class_<MPI_Utils>(m, "MPI_Utils")    
        .def(py::init<>())
        .def("reconcile_gather", &MPI_Utils::reconcile_gather, "reconcile_gather")
        .def("reconcile_dp_matrix", &MPI_Utils::reconcile_dp_matrix, "reconcile_dp_matrix")
        .def("reconcile_dp_optimized", &MPI_Utils::reconcile_dp_optimized, "reconcile_dp_matrix");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}