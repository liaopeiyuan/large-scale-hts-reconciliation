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

int add(int i, int j) {
    return i + j;
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
  
  void say_hi() {
    int world_size;
    MPI_Comm_size(comm_global, &world_size);
    int world_rank;
    MPI_Comm_rank(comm_global, &world_rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME] = "localhost";
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("[C++]    Hello from machine %s, MPI rank %d out of %d\n",
	   processor_name,
	   world_rank,
	   world_size);
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

    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");

    m.def("inv", &inv);
    m.def("det", &det);

    py::class_<Distributed>(m, "Distributed")    
        .def(py::init<>())
        .def("say_hi", &Distributed::say_hi, "Each process will say hi");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}