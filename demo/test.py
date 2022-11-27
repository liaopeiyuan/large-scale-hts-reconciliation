#!/usr/bin/env python3

import mpi4py

mpi4py.rc.threaded = True
mpi4py.rc.thread_level = "funneled"
from mpi4py import MPI
from lhts import Distributed
import lhts
import numpy as np

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    name = MPI.Get_processor_name()
    A = np.randn((3, 3))
    print(name + "," + str(rank) + "," + str(size))
    distrib = Distributed()
    distrib.say_hi(A)
    print(lhts.det(A))


if __name__ == "__main__":
    main()
