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

    start = timer()
    #if (rank == size - 1):
    S_compact = np.load(open(data_dir + 'm5_hierarchy_parent.npy', 'rb'))
    top_down_p = np.load(open(data_dir + 'm5_prediction_raw/top_down_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)
    level_2_p = np.load(open(data_dir + 'm5_prediction_raw/level_2_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)

    #else:
    #    S_compact, top_down_p = np.array([]), np.array([])
    
    gt = np.load(open(data_dir + 'm5_prediction_raw/mpi/gt_tensor_' + str(rank) + '.npy', 'rb'))
    yhat = np.load(open(data_dir + 'm5_prediction_raw/mpi/pred_tensor_' + str(rank) + '.npy', 'rb'))
    
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == size - 1): 
        print("Load: " + str(elapsed))
        print(gt[-5:, :])

    start = timer()
    rec = distrib.reconcile_dp_matrix("top_down", S_compact, top_down_p, yhat, -1, 0.0, 5650, 6218, 4)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == size - 1): 
        print("Top down (dp matrix): ", str(elapsed), " ", lhts.smape(rec, gt))
        #print(rec[-5:, :])

    start = timer()
    rec2 = distrib.reconcile_dp_optimized("top_down", S_compact, top_down_p, yhat, -1, 0.0, 5650, 6218, 4)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == size - 1): 
        print("Top down (dp algo): ", str(elapsed), " ", lhts.smape(rec2, gt))

    start = timer()
    rec = distrib.reconcile_dp_matrix("bottom_up", S_compact, top_down_p, yhat, -1, 0.0, 5650, 6218, 4)
    end = timer()
    elapsed = round(end - start, 4)
    #if (rank == size - 1): 
    if (rank >= 10): print("Bottom up (dp matrix): ", rank, " ", str(elapsed), " ", lhts.smape(rec, gt))

    start = timer()
    rec2 = distrib.reconcile_dp_optimized("bottom_up", S_compact, top_down_p, yhat, -1, 0.0, 5650, 6218, 4)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank >= 10): 
        print("Bottom up (dp algo): ", rank, " ", str(elapsed), " ", lhts.smape(rec2, gt))
        print(rank, np.abs(rec[:, :] - rec2[:, :]).sum(), np.abs(yhat[:, :] - rec2[:, :]).sum(), "\n")
    
    start = timer()
    rec = distrib.reconcile_dp_matrix("middle_out", S_compact, level_2_p, yhat, 2, 0.0, 5650, 6218, 4)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == size - 1):
        print("Middle out: ", str(elapsed), " ", lhts.smape(rec, gt))
        #print(rec[-5:, :])

    start = timer()
    rec2 = distrib.reconcile_dp_optimized("middle_out", S_compact, level_2_p, yhat, 2, 0.0, 5650, 6218, 4)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == size - 1): 
        print("Middle out (dp algo): ", str(elapsed), " ", lhts.smape(rec2, gt))

    start = timer()
    rec = distrib.reconcile_dp_matrix("OLS", S_compact, top_down_p, yhat, 2, 0.0, 5650, 6218, 4)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == size - 1):
        print("OLS: ", str(elapsed), " ", lhts.smape(rec, gt))
        print(rec[-5:, :])

    start = timer()
    rec = distrib.reconcile_dp_matrix("WLS", S_compact, top_down_p, yhat, 2, 0.5, 5650, 6218, 4)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == size - 1):
        print("WLS: ", str(elapsed), " ", lhts.smape(rec, gt))
        print(rec[-5:, :])
    

if __name__ == "__main__":
    main()