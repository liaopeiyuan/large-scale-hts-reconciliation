#!/usr/bin/env python3

import mpi4py

mpi4py.rc.threaded = True
mpi4py.rc.thread_level = "funneled"
from mpi4py import MPI
from lhts import Distributed
import lhts

import numpy as np
from timeit import default_timer as timer

def main():
    ROOT = "/data/cmu/large-scale-hts-reconciliation/"
    data_dir = ROOT + "notebooks/"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    distrib = Distributed()

    S_compact = np.load(open(data_dir + 'm5_hierarchy_parent.npy', 'rb'))
    y_hat = np.load(open(data_dir + 'm5_prediction_raw/mpi/pred_tensor_' + str(rank) + '.npy', 'rb'))[:, 0].reshape(-1, 1)
    top_down_p = np.load(open(data_dir + 'm5_prediction_raw/top_down_tensor.npy', 'rb'))
    print(y_hat.shape)

    start = timer()
    print(distrib.reconcile_naive_mpi("bottom_up", S_compact, top_down_p, y_hat, -1, 0.0, 5650, 6218, 4).shape)
    end = timer()
    elapsed = round(end - start, 4)
    print("Bottom up: " + str(elapsed))

    start = timer()
    print(distrib.reconcile_naive_mpi("top_down", S_compact, top_down_p, y_hat, -1, 0.0, 5650, 6218, 4).shape)
    end = timer()
    elapsed = round(end - start, 4)
    print("Top down: " + str(elapsed))

    start = timer()
    print(distrib.reconcile_naive_mpi("middle_out", S_compact, top_down_p, y_hat, 2, 0.0, 5650, 6218, 4).shape)
    end = timer()
    elapsed = round(end - start, 4)
    print("Middle out: " + str(elapsed))

    start = timer()
    print(distrib.reconcile_naive_mpi("OLS", S_compact, top_down_p, y_hat, 2, 0.0, 5650, 6218, 4).shape)
    end = timer()
    elapsed = round(end - start, 4)
    print("OLS: " + str(elapsed))

    start = timer()
    print(distrib.reconcile_naive_mpi("WLS", S_compact, top_down_p, y_hat, 2, 0.5, 5650, 6218, 4).shape)
    end = timer()
    elapsed = round(end - start, 4)
    print("WLS: " + str(elapsed))

    

if __name__ == "__main__":
    main()