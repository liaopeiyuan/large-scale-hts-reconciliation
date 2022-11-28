// Referencing https://github.com/latug0/pybind_mpi/

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <vector>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>
#include <stdio.h>
#include <Eigen/LU>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

Eigen::MatrixXf reconcile_matrix(const Eigen::MatrixXf G, const Eigen::MatrixXf S, const Eigen::MatrixXf yhat) {
    Eigen::MatrixXf res = S * G;
    res = res * yhat;
    return res;
}

Eigen::MatrixXi construct_S(const Eigen::MatrixXi S_compact, int num_base, int num_total, int num_levels) {
    Eigen::MatrixXi S = Eigen::MatrixXi::Zero(num_total, num_base);
    
    assert(S_compact.rows() == num_total);
    assert(S_compact.cols() == num_levels);
    assert(num_levels > 1);

    #pragma omp parallel for 
    for (int i = 0; i < num_total; i++) {
        int co = S_compact(i, 0);
        for (int j = 1; j < num_levels; j++) {
            int ro = S_compact(i, j);
            if (ro == -1) {
                break;
            } else {
                if (co < num_base) S(ro, co) = 1;
            }
        }
    }

    return S;
}

Eigen::MatrixXf construct_G_OLS(const Eigen::MatrixXi S) {
    Eigen::MatrixXf Sp = S.cast<float>();
    Eigen::MatrixXf St = Sp.transpose();
    return (St * Sp).inverse() * St;
}

Eigen::MatrixXf construct_G_WLS(const Eigen::MatrixXi S, int w) {
    Eigen::MatrixXf W = Eigen::MatrixXf::Zero(S.rows(), S.cols());
    #pragma omp parallel for 
    for (int i = 0; i < S.rows(); i++) {
        W(i, i) = w;
    }
    Eigen::MatrixXf Sp = S.cast<float>();
    Eigen::MatrixXf St = Sp.transpose();
    return (St * W * Sp).inverse() * St * W;
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
        int max_id = -1;
        bool is_base = true;
        int _level = 1;
        for (int j = 1; j < num_levels; j++) {
            int ro = S_compact(i, j);
            if (ro != -1) {
                level++;
                max_id = ro;
            }
        }
        if (_level == level) {
            G(co, max_id) = P(co, 0);
        }
    }

    return G;
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
        int max_id = -1;
        bool is_base = true;
        for (int j = 1; j < num_levels; j++) {
            int ro = S_compact(i, j);
            if (ro == -1) {
                is_base = false;
                break;
            }
            max_id = ro;
        }
        if (is_base) {
            G(co, max_id) = P(co, 0);
        }
    }

    return G;
}

Eigen::MatrixXi construct_G_bottom_up(const Eigen::MatrixXi S_compact, int num_base, int num_total, int num_levels) {
    Eigen::MatrixXi G = Eigen::MatrixXi::Zero(num_base, num_total);
    
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
            G(i, i) = 1;
        }
    }

    return G;
}

Eigen::MatrixXf reconcile(const std::string method,
                          const Eigen::MatrixXi S_compact,
                          const Eigen::MatrixXf yhat,
                          int num_base, int num_total, int num_levels) {
    switch (method) {
        case "bottom_up":
        break;
        case "top_down":
        break;
        case "middle_out":
        break;
        case "OLS":
        break;
        case "WLS":
        break;
        default:
            throw std::invalid_argument("invalid reconciliation method");
        break;
    }
    Eigen::MatrixXi S = construct_S(S_compact, num_base, num_total, num_levels);
    printf("S\n");
    Eigen::MatrixXi G = construct_G_bottom_up(S_compact, num_base, num_total, num_levels);
    printf("G\n");
    Eigen::MatrixXf res = (S * G).cast<float>();
    printf("res\n");
    res = res * yhat;
    return res;
}

Eigen::MatrixXd inv(const Eigen::MatrixXd &xs)
{
  return xs.inverse();
}

double det(const Eigen::MatrixXd &xs)
{
  return xs.determinant();
}

namespace py = pybind11;
using pymod = pybind11::module;

class Distributed
{
public:
  Distributed() : comm_global(MPI_COMM_WORLD) {}
  
  ~Distributed() {}
  
  Eigen::MatrixXf reconcile_naive_mpi(const Eigen::MatrixXi S_compact,
                                      const Eigen::MatrixXf yhat, int num_base, int num_total, int num_levels) {
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

    if (world_rank == 0) {
        yhat_total = Eigen::MatrixXf::Zero(std::accumulate(rows.begin(), rows.end(), 0), 
                                           cols[0]);
        std::vector<Eigen::MatrixXf> yhats(world_size);

        for (int i = 1; i < world_size; i++) {
            yhats[i] = Eigen::MatrixXf::Zero(rows[i], cols[i]);
            MPI_Irecv(yhats[i].data(), rows[i] * cols[i], MPI_FLOAT, i, 0, comm_global, &reqs[i]);
        }

        MPI_Waitall(world_size, reqs.data(), stats.data());

        int curr_row = rows[0];

        yhat_total.topRows(rows[0]) = yhat;

        for (int i = 1; i < world_size; i++) {
            yhat_total.middleRows(curr_row, rows[i]) = yhats[i];
            curr_row += rows[i];
        }

    } else {
        MPI_Isend(yhat.data(), ro * co, MPI_FLOAT, 0, 0, comm_global, &reqs[0]);
        MPI_Wait(&reqs[0], &stats[0]);
    }

    if (world_rank == 0) {
        Eigen::MatrixXi S = construct_S(S_compact, num_base, num_total, num_levels);
        Eigen::MatrixXi G = construct_G_bottom_up(S_compact, num_base, num_total, num_levels);
        Eigen::MatrixXf res = (S * G).cast<float>();
        res = res * yhat_total;
        return res;
    } else {
        return yhat;
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

    m.def("inv", &inv);
    m.def("det", &det);

    m.def("reconcile_matrix", &reconcile_matrix);
    m.def("reconcile", &reconcile);
    m.def("construct_S", &construct_S);
    m.def("construct_G_bottom_up", &construct_G_bottom_up);
    m.def("construct_G_top_down", &construct_G_top_down);
    m.def("construct_G_middle_out", &construct_G_middle_out);

    py::class_<Distributed>(m, "Distributed")    
        .def(py::init<>())
        .def("test", &Distributed::test, "test");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}