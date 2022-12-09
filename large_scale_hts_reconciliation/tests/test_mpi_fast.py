import lhts
import numpy as np
import pytest
import itertools
from collections import defaultdict
import mpi4py
mpi4py.rc.threaded = True
mpi4py.rc.thread_level = "funneled"
from mpi4py import MPI
from lhts import MPI_utils
import sys
import numpy as np
from timeit import default_timer as timer

ROOT = "/data/cmu/large-scale-hts-reconciliation/"
data_dir = ROOT + "notebooks/"

METHOD = "middle_out"
S_compact = np.load(open(data_dir + 'm5_hierarchy_parent.npy', 'rb'))
    
if (METHOD == "middle_out"):
    P = np.load(open(data_dir + 'm5_prediction_raw/top_down_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)
else:
    P = np.load(open(data_dir + 'm5_prediction_raw/level_2_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)

@pytest.mark.mpi
def test_mpi():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    distrib = MPI_utils()

    with gt as np.load(open(data_dir + 'm5_prediction_raw/mpi/gt_tensor_' + str(rank) + '.npy', 'rb')):
        with yhat as np.load(open(data_dir + 'm5_prediction_raw/mpi/pred_tensor_' + str(rank) + '.npy', 'rb')):
            pass
        
    #gt = np.load(open(data_dir + 'm5_prediction_raw/mpi/gt_tensor_' + str(rank) + '.npy', 'rb'))
    #yhat = np.load(open(data_dir + 'm5_prediction_raw/mpi/pred_tensor_' + str(rank) + '.npy', 'rb'))

    return

    start = timer()
    rec = distrib.reconcile_dp_matrix(METHOD, S_compact, 30490, 33549, 4, yhat, P, 2, 0.0)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == size - 1):
        print(METHOD, ":")
        print("dp matrix: ", str(elapsed), " ", lhts.smape(rec, gt))

    start = timer()
    rec2 = distrib.reconcile_dp_optimized(METHOD, S_compact, 30490, 33549, 4, yhat, P, 2, 0.0)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == size - 1): 
        print("dp algo: ", str(elapsed), " ", lhts.smape(rec2, gt))

    start = timer()
    rec3 = distrib.reconcile_gather(METHOD, S_compact, 30490, 33549, 4, yhat, P, 2, 0.0)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == size - 1):
        print("gather: ", str(elapsed), " ", lhts.smape(rec3, gt))
        print("dp mat vs dp algo: ", np.abs(rec2 - rec).sum())
        print("gather vs dp algo: ", np.abs(rec2- rec3).sum())
        print("gather vs dp mat: ", np.abs(rec - rec3).sum())