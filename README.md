# Large-Scale Hierarchical Time-Series Reconciliation

[hts.ml.works](http://hts.ml.works)


## Building

1. Install [Eigen 3](https://eigen.tuxfamily.org/index.php?title=Main_Page) for OpenMP / LAPACK / BLAS support on matrix operations
   * use `-DCMAKE_INSTALL_PREFIX=` to install locally
2. Clone this repository with pybind11 submodule
3. `python -m pip install large_scale_hts_reconciliation/` to install the pip package with CMake
4. Run MPI benchmark with `mpirun -n 12 --bind-to none python3 demo/benchmark_reconcile_naive_mpi.py`