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

methods = ["OLS", "WLS"]
modes = ["dense_algo", "sparse_algo", "dense_matrix", "sparse_matrix"]

def run(mode, method):
    if mode == "sparse_matrix":
        return lambda: lhts.reconcile_sparse_matrix(method, S_compact, 5650, 6218, 4, yhat, top_down_p, -1, 1.5)
    elif mode == "sparse_algo":
        return lambda: lhts.reconcile_sparse_algo(method, S_compact, 5650, 6218, 4, yhat, top_down_p, -1, 1.5)
    elif mode == "dense_matrix":
        return lambda: lhts.reconcile_dense_matrix(method, S_compact, 5650, 6218, 4, yhat, top_down_p, -1, 1.5)
    elif mode == "dense_algo":
        return lambda: lhts.reconcile_dense_algo(method, S_compact, 5650, 6218, 4, yhat, top_down_p, -1, 1.5)

d = defaultdict(dict)

@pytest.mark.parametrize(
    "mode,method", itertools.product(modes, methods)
)
def test_single_process(benchmark, mode, method):
    benchmark.group = method 
    result = run(mode, method)
    
    d[method][mode] = result
    for (i, j) in itertools.combinations(d[method].values(), 2):
        assert np.allclose(i, j, rtol=1e-3, atol=1e-5)
