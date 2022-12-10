import lhts
import numpy as np
import pytest
import itertools
from collections import defaultdict
import mpi4py
mpi4py.rc.threaded = True
mpi4py.rc.thread_level = "funneled"
from mpi4py import MPI
from lhts import Distributed
import sys
import numpy as np
from timeit import default_timer as timer

ROOT = "/data/cmu/large-scale-hts-reconciliation/"
data_dir = ROOT + "notebooks/"

S_compact = np.load(open(data_dir + 'm5_hierarchy_parent.npy', 'rb'))    
top_down_p = np.load(open(data_dir + 'm5_prediction_raw/top_down_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)
level_2_p = np.load(open(data_dir + 'm5_prediction_raw/level_2_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

gt = np.load(open(data_dir + 'm5_prediction_raw/mpi/gt_tensor_' + str(rank) + '.npy', 'rb'))
yhat = np.load(open(data_dir + 'm5_prediction_raw/mpi/pred_tensor_' + str(rank) + '.npy', 'rb'))

methods = ["bottom_up"] #, "middle_out", "top_down"]
modes = ["dp_matrix", "dp_optimized"]

def run_bottom_up(distrib, mode):
    if mode == "gather":
        return lambda: distrib.reconcile_gather("bottom_up", S_compact, 30490, 33549, 4, yhat, top_down_p, -1, 1.5)
    elif mode == "dp_matrix":
        return lambda: distrib.reconcile_dp_matrix("bottom_up", S_compact, 30490, 33549, 4, yhat, top_down_p, -1, 1.5)
    elif mode == "dp_optimized":
        return lambda: distrib.reconcile_dp_optimized("bottom_up", S_compact, 30490, 33549, 4, yhat, top_down_p, -1, 1.5)

def run_top_down(distrib, mode):
    if mode == "gather":
        return lambda: distrib.reconcile_gather("top_down", S_compact, 30490, 33549, 4, yhat, top_down_p, -1, 1.5)
    elif mode == "dp_matrix":
        return lambda: distrib.reconcile_dp_matrix("top_down", S_compact, 30490, 33549, 4, yhat, top_down_p, -1, 1.5)
    elif mode == "dp_optimized":
        return lambda: distrib.reconcile_dp_optimized("top_down", S_compact, 30490, 33549, 4, yhat, top_down_p, -1, 1.5)

def run_middle_out(distrib, mode):
    if mode == "gather":
        return lambda: distrib.reconcile_gather("middle_out", S_compact, 30490, 33549, 4, yhat, level_2_p, 2, 1.5)
    elif mode == "dp_matrix":
        return lambda: distrib.reconcile_dp_matrix("middle_out", S_compact, 30490, 33549, 4, yhat, level_2_p, 2, 1.5)
    elif mode == "dp_optimized":
        return lambda: distrib.reconcile_dp_optimized("middle_out", S_compact, 30490, 33549, 4, yhat, level_2_p, 2, 1.5)
 
d = defaultdict(dict)

@pytest.mark.mpi()
@pytest.mark.parametrize(
    "mode,method", itertools.product(modes, methods)
)
@pytest.mark.benchmark(
    min_rounds=1,
)
def test_mpi(benchmark, mode, method):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    distrib = Distributed()

    benchmark.group = method 
    
    if method == "bottom_up":
        result = benchmark(run_bottom_up(distrib, mode))
    elif method == "middle_out":
        result = benchmark(run_middle_out(distrib, mode))
    elif method == "top_down":
        result = benchmark(run_top_down(distrib, mode))
    
    d[method][mode] = result
    for (i, j) in itertools.combinations(d[method].values(), 2):
        assert np.allclose(i, j, rtol=1e-3, atol=1e-5)
