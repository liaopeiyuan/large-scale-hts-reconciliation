// Referencing https://github.com/latug0/pybind_mpi/

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>
#include <stdio.h>
#include <Eigen/LU>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

Eigen::MatrixXf reconcile(const Eigen::MatrixXf G, const Eigen::MatrixXf S, const Eigen::MatrixXf yhat) {
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

Eigen::MatrixXi construct_G_OLS(const Eigen::MatrixXi S) {
    Eigen::MatrixXi St = S.transpose();
    return (St * S).inverse() * St;
}

Eigen::MatrixXi construct_G_WLS(const Eigen::MatrixXi S, int w) {
    Eigen::MatrixXi W = Eigen::MatrixXi::Zero(S.rows(), S.cols());
    #pragma omp parallel for 
    for (int i = 0; i < min(S.rows(), S.cols()); i++) {
        W(i, i) = w;
    }
    Eigen::MatrixXi St = S.transpose();
    return (St * W * S).inverse() * St * W;
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

Eigen::MatrixXf reconcile_bottom_up(const Eigen::MatrixXi S_compact,
                                    const Eigen::MatrixXf yhat, int num_base, int num_total, int num_levels) {
    Eigen::MatrixXi S = construct_S(S_compact, num_base, num_total, num_levels);
    Eigen::MatrixXi G = construct_G_bottom_up(S_compact, num_base, num_total, num_levels);
    Eigen::MatrixXf res = (S * G).cast<float>();
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

    m.def("reconcile", &reconcile);
    m.def("reconcile_bottom_up", &reconcile_bottom_up);
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