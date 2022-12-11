# Large-Scale Hierarchical Time-Series Reconciliation

[hts.ml.works](http://hts.ml.works)


## Building

1. Install [Eigen 3](https://eigen.tuxfamily.org/index.php?title=Main_Page) for OpenMP / LAPACK / BLAS support on matrix operations
   * use `-DCMAKE_INSTALL_PREFIX=` to install locally
   * `export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:{YOUR_PATH_TO_EIGEN3}"` to make sure CMake functions properly
2. Clone this repository with pybind11 submodule
3. Install requirements in `requirements.txt`
   * Remark our own fork of `pytest-mpi` that is capable of producing the stdout of a single process while suppressing others.
4. `python -m pip install large_scale_hts_reconciliation/` to install the pip package with CMake


## Reproducing Benchmarking Results

1. Allocate two GCP `e2-highcpu-16` instances with 1 vCPU per core, 8 cores and Intel Broadwell x86/64
platform/architecture. One instance is in `us-central1-a` and the other is in `us-east1-b1`.
2. Install OpenMPI 4.1.4 and set up `hosts` file properly (see passwordless SSH)
3. For single-node results, run `mpirun -n 8 --bind-to socket python3 -m pytest --with-mpi $HOME/large-scale-hts-reconciliation/large_scale_hts_reconciliation/tests/test_mpi_fast.py --unmute-ranks=0` and `mpirun -n 1 --bind-to socket python3 -m pytest tests/test_single_process_fast.py` for multi-process benchmarks (test_mpi_fast.py) and single-process benchmarks (test_single_process_fast.py)
4. For multi-node results, run `mpirun --hostfile hosts -n 8 --bind-to socket python3 -m pytest --with-mpi $HOME/large-scale-hts-reconciliation/large_scale_hts_reconciliation/tests/test_mpi_fast.py --unmute-ranks=0`