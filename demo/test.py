#!/usr/bin/env python3

import mpi4py

mpi4py.rc.threaded = True
mpi4py.rc.thread_level = "funneled"
from mpi4py import MPI
from lhts import MPI_Utils
import lhts
import numpy as np

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    name = MPI.Get_processor_name()
    A = np.random.normal(size=(3, 3))
    print(str(rank) + "," + str(size) + ",", lhts.det(A))
    distrib = MPI_Utils()
    distrib.test(A)


if __name__ == "__main__":
    main()
