#!/usr/bin/env python3

import mpi4py

mpi4py.rc.threaded = True
mpi4py.rc.thread_level = "funneled"
from mpi4py import MPI
from lhts import MPI_utils
import lhts

import numpy as np
from timeit import default_timer as timer


def main():
    ROOT = "/data/cmu/large-scale-hts-reconciliation/"
    data_dir = ROOT + "notebooks/"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    distrib = MPI_utils()

    RANK_TO_TEST = size - 1

    start = timer()
    #if (rank == 0):
    S_compact = np.load(open(data_dir + 'm5_hierarchy_parent.npy', 'rb'))
    top_down_p = np.load(open(data_dir + 'm5_prediction_raw/top_down_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)
    level_2_p = np.load(open(data_dir + 'm5_prediction_raw/level_2_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)

    #else:
    #    S_compact, top_down_p = np.array([]), np.array([])
    
    gt = np.load(open(data_dir + 'm5_prediction_raw/mpi/gt_tensor_' + str(rank) + '.npy', 'rb'))
    yhat = np.load(open(data_dir + 'm5_prediction_raw/mpi/pred_tensor_' + str(rank) + '.npy', 'rb'))
    
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == RANK_TO_TEST): 
        print("Original: ", str(elapsed), " ", lhts.smape(yhat, gt))
        print(yhat[-5:, -1])

    #if (rank == 0): print(S_compact.shape, top_down_p.shape, yhat.shape)    
    start = timer()
    rec = distrib.reconcile_gather("top_down", S_compact, 5650, 6218, 4, yhat, top_down_p)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == RANK_TO_TEST): 
        print("Top down: ", str(elapsed), " ", lhts.smape(rec, gt))
        print(rec[-5:, -1])

    #if (rank == 0): print(S_compact.shape, top_down_p.shape, yhat.shape)    
    start = timer()
    rec = distrib.reconcile_gather("bottom_up", S_compact, 5650, 6218, 4, yhat)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == RANK_TO_TEST): 
        print("Bottom up: ", str(elapsed), " ", lhts.smape(rec, gt))
        print(rec[-5:, -1])


    start = timer()
    rec = distrib.reconcile_gather("middle_out", S_compact, 5650, 6218, 4, yhat, level_2_p, 2)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == RANK_TO_TEST):
        print("Middle out: ", str(elapsed), " ", lhts.smape(rec, gt))
        print(rec[-5:, -1])

    start = timer()
    rec = distrib.reconcile_gather("OLS", S_compact, 5650, 6218, 4, yhat)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == RANK_TO_TEST):
        print("OLS: ", str(elapsed), " ", lhts.smape(rec, gt))
        print(rec[-5:, -1])

    start = timer()
    rec = distrib.reconcile_gather("WLS", S_compact, 5650, 6218, 4, yhat, top_down_p, 0, 1.5)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == RANK_TO_TEST):
        print("WLS: ", str(elapsed), " ", lhts.smape(rec, gt))
        print(rec[-5:, -1])

    

if __name__ == "__main__":
    main()