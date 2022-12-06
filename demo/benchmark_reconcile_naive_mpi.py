#!/usr/bin/env python3

import mpi4py

mpi4py.rc.threaded = True
mpi4py.rc.thread_level = "funneled"
from mpi4py import MPI
from lhts import MPI_Utils
import lhts

import numpy as np
from timeit import default_timer as timer

def main():
    ROOT = "/data/cmu/large-scale-hts-reconciliation/"
    data_dir = ROOT + "notebooks/"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    distrib = MPI_Utils()

    start = timer()
    #if (rank == 0):
    S_compact = np.load(open(data_dir + 'm5_hierarchy_parent.npy', 'rb'))
    top_down_p = np.load(open(data_dir + 'm5_prediction_raw/top_down_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)
    level_2_p = np.load(open(data_dir + 'm5_prediction_raw/level_2_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)

    #else:
    #    S_compact, top_down_p = np.array([]), np.array([])
    
    gt = np.load(open(data_dir + 'm5_prediction_raw/mpi/gt_tensor_' + str(rank) + '.npy', 'rb'))[:, 0].reshape(-1, 1)
    y_hat = np.load(open(data_dir + 'm5_prediction_raw/mpi/pred_tensor_' + str(rank) + '.npy', 'rb'))[:, 0].reshape(-1, 1)
    
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == size - 1): 
        print("Load: " + str(elapsed))
        print(gt)

    #if (rank == 0): print(S_compact.shape, top_down_p.shape, y_hat.shape)    
    start = timer()
    rec = distrib.reconcile_naive("top_down", S_compact, top_down_p, y_hat, -1, 0.0, 5650, 6218, 4)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == size - 1): 
        print("Top down: ", str(elapsed), " ", lhts.smape(rec, gt))
        print(rec)

    #if (rank == 0): print(S_compact.shape, top_down_p.shape, y_hat.shape)    
    start = timer()
    rec = distrib.reconcile_naive("bottom_up", S_compact, top_down_p, y_hat, -1, 0.0, 5650, 6218, 4)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == size - 1): 
        print("Bottom up: ", str(elapsed), " ", lhts.smape(rec, gt))
        print(rec)


    start = timer()
    rec = distrib.reconcile_naive("middle_out", S_compact, top_down_p, y_hat, 2, 0.0, 5650, 6218, 4)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == size - 1):
        print("Middle out: ", str(elapsed), " ", lhts.smape(rec, gt))
        print(rec)

    start = timer()
    rec = distrib.reconcile_naive("OLS", S_compact, top_down_p, y_hat, 2, 0.0, 5650, 6218, 4)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == 0):
        print("OLS: ", str(elapsed), " ", lhts.smape(rec, gt))
        print(rec)

    start = timer()
    rec = distrib.reconcile_naive("WLS", S_compact, top_down_p, y_hat, 2, 0.5, 5650, 6218, 4)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == 0):
        print("WLS: ", str(elapsed), " ", lhts.smape(rec, gt))
        print(rec)

    

if __name__ == "__main__":
    main()