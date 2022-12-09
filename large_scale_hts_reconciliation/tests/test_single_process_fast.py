import lhts
import numpy as np
import pytest
import itertools
from collections import defaultdict

ROOT = "/data/cmu/large-scale-hts-reconciliation/"
data_dir = ROOT + "notebooks/"

S_compact = np.load(open(data_dir + 'm5_hierarchy_parent.npy', 'rb'))
yhat = np.load(open(data_dir + 'm5_prediction_raw/pred_tensor.npy', 'rb'))
gt = np.load(open(data_dir + 'm5_prediction_raw/gt_tensor.npy', 'rb'))
top_down_p = np.load(open(data_dir + 'm5_prediction_raw/top_down_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)
level_2_p = np.load(open(data_dir + 'm5_prediction_raw/level_2_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)

methods = ["middle_out", "bottom_up", "top_down"]
modes = ["dense_algo", "sparse_algo", "dense_matrix", "sparse_matrix"]

def run_bottom_up(mode):
    if mode == "sparse_matrix":
        return lambda: lhts.reconcile_sparse_matrix("bottom_up", S_compact, 30490, 33549, 4, yhat, top_down_p, -1, 0.0)
    elif mode == "sparse_algo":
        return lambda: lhts.reconcile_sparse_algo("bottom_up", S_compact, 30490, 33549, 4, yhat, top_down_p, -1, 0.0)
    elif mode == "dense_matrix":
        return lambda: lhts.reconcile_dense_matrix("bottom_up", S_compact, 30490, 33549, 4, yhat, top_down_p, -1, 0.0)
    elif mode == "dense_algo":
        return lambda: lhts.reconcile_dense_algo("bottom_up", S_compact, 30490, 33549, 4, yhat, top_down_p, -1, 0.0)

def run_top_down(mode):
    if mode == "sparse_matrix":
        return lambda: lhts.reconcile_sparse_matrix("top_down", S_compact, 30490, 33549, 4, yhat, top_down_p, -1, 0.0)
    elif mode == "sparse_algo":
        return lambda: lhts.reconcile_sparse_algo("top_down", S_compact, 30490, 33549, 4, yhat, top_down_p, -1, 0.0)
    elif mode == "dense_matrix":
        return lambda: lhts.reconcile_dense_matrix("top_down", S_compact, 30490, 33549, 4, yhat, top_down_p, -1, 0.0)
    elif mode == "dense_algo":
        return lambda: lhts.reconcile_dense_algo("top_down", S_compact, 30490, 33549, 4, yhat, top_down_p, -1, 0.0)

def run_middle_out(mode):
    if mode == "sparse_matrix":
        return lambda: lhts.reconcile_sparse_matrix("middle_out", S_compact, 30490, 33549, 4, yhat, level_2_p, 2, 0.0)
    elif mode == "sparse_algo":
        return lambda: lhts.reconcile_sparse_algo("middle_out", S_compact, 30490, 33549, 4, yhat, level_2_p, 2, 0.0)
    elif mode == "dense_matrix":
        return lambda: lhts.reconcile_dense_matrix("middle_out", S_compact, 30490, 33549, 4, yhat, level_2_p, 2, 0.0)
    elif mode == "dense_algo":
        return lambda: lhts.reconcile_dense_algo("middle_out", S_compact, 30490, 33549, 4, yhat, level_2_p, 2, 0.0)

d = defaultdict(dict)

@pytest.mark.parametrize(
    "mode,method", itertools.product(modes, methods)
)
def test_single_process_fast(benchmark, mode, method):
    benchmark.group = method 
    
    if method == "bottom_up":
        result = benchmark(run_bottom_up(mode))
    elif method == "middle_out":
        result = benchmark(run_middle_out(mode))
    elif method == "top_down":
        result = benchmark(run_top_down(mode))

    d[method][mode] = result
    for (i, j) in itertools.combinations(d[method].values(), 2):
        assert np.allclose(i, j, rtol=1e-3, atol=1e-5)
