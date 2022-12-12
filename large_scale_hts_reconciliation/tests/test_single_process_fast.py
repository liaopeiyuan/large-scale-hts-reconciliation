import lhts
import numpy as np
import pytest
import itertools
from collections import defaultdict

DATASETS = ["m5_hobbies", "wikipedia", "m5_full"]

hierarchy_prefix = {"m5_hobbies": "m5", "m5_full": "m5", "wikipedia": "wikipedia"}
num_leaves = {"m5_hobbies": 5650, "m5_full": 30490}
num_nodes = {"m5_hobbies": 6218, "m5_full": 33549}
num_levels = {"m5_hobbies": 4, "m5_full": 4}


ROOT = "/home/peiyuan20013/large-scale-hts-reconciliation/large_scale_hts_reconciliation"  # "/data/cmu/large-scale-hts-reconciliation/"
data_dir = ROOT + "/notebooks/"

S_compacts = {}
top_down_ps = {}
level_2_ps = {}
gts = {}
yhats = {}

for DATA_ROOT in DATASETS:
    S_compact = np.load(open(data_dir + DATA_ROOT + "/" + hierarchy_prefix[DATA_ROOT] + "_hierarchy_parent.npy", "rb"))
    top_down_p = np.load(open(data_dir + DATA_ROOT + "/top_down_tensor.npy", "rb"))[
        :, 0
    ].reshape(-1, 1)
    level_2_p = np.load(open(data_dir + DATA_ROOT + "/level_2_tensor.npy", "rb"))[
        :, 0
    ].reshape(-1, 1)

    yhat = np.load(open(data_dir + DATA_ROOT + "/pred_tensor.npy", "rb"))
    gt = np.load(open(data_dir + DATA_ROOT + "/gt_tensor.npy", "rb"))

    S_compacts[DATA_ROOT] = S_compact
    top_down_ps[DATA_ROOT] = top_down_p
    level_2_ps[DATA_ROOT] = level_2_p
    gts[DATA_ROOT] = gt
    yhats[DATA_ROOT] = yhat

methods = ["middle_out", "bottom_up", "top_down"]
modes = ["sparse_matrix", "sparse_algo"] #, "dense_matrix", "dense_algo"]

def run_bottom_up(mode, dataset):
    if mode == "sparse_matrix":
        return lambda: lhts.reconcile_sparse_matrix(
            "bottom_up", S_compacts[dataset], num_leaves[dataset], num_nodes[dataset], num_levels[dataset], yhats[dataset], top_down_ps[dataset], -1, 0.0
        )
    elif mode == "sparse_algo":
        return lambda: lhts.reconcile_sparse_algo(
            "bottom_up", S_compacts[dataset], num_leaves[dataset], num_nodes[dataset], num_levels[dataset], yhats[dataset], top_down_ps[dataset], -1, 0.0
        )
    elif mode == "dense_matrix":
        return lambda: lhts.reconcile_dense_matrix(
            "bottom_up", S_compacts[dataset], num_leaves[dataset], num_nodes[dataset], num_levels[dataset], yhats[dataset], top_down_ps[dataset], -1, 0.0
        )
    elif mode == "dense_algo":
        return lambda: lhts.reconcile_dense_algo(
            "bottom_up", S_compacts[dataset], num_leaves[dataset], num_nodes[dataset], num_levels[dataset], yhats[dataset], top_down_ps[dataset], -1, 0.0
        )


def run_top_down(mode, dataset):
    if mode == "sparse_matrix":
        return lambda: lhts.reconcile_sparse_matrix(
            "top_down", S_compacts[dataset], num_leaves[dataset], num_nodes[dataset], num_levels[dataset], yhats[dataset], top_down_ps[dataset], -1, 0.0
        )
    elif mode == "sparse_algo":
        return lambda: lhts.reconcile_sparse_algo(
            "top_down", S_compacts[dataset], num_leaves[dataset], num_nodes[dataset], num_levels[dataset], yhats[dataset], top_down_ps[dataset], -1, 0.0
        )
    elif mode == "dense_matrix":
        return lambda: lhts.reconcile_dense_matrix(
            "top_down", S_compacts[dataset], num_leaves[dataset], num_nodes[dataset], num_levels[dataset], yhats[dataset], top_down_ps[dataset], -1, 0.0
        )
    elif mode == "dense_algo":
        return lambda: lhts.reconcile_dense_algo(
            "top_down", S_compacts[dataset], num_leaves[dataset], num_nodes[dataset], num_levels[dataset], yhats[dataset], top_down_ps[dataset], -1, 0.0
        )


def run_middle_out(mode, dataset):
    if mode == "sparse_matrix":
        return lambda: lhts.reconcile_sparse_matrix(
            "middle_out", S_compacts[dataset], num_leaves[dataset], num_nodes[dataset], num_levels[dataset], yhats[dataset], level_2_ps[dataset], 2, 0.0
        )
    elif mode == "sparse_algo":
        return lambda: lhts.reconcile_sparse_algo(
            "middle_out", S_compacts[dataset], num_leaves[dataset], num_nodes[dataset], num_levels[dataset], yhats[dataset], level_2_ps[dataset], 2, 0.0
        )
    elif mode == "dense_matrix":
        return lambda: lhts.reconcile_dense_matrix(
            "middle_out", S_compacts[dataset], num_leaves[dataset], num_nodes[dataset], num_levels[dataset], yhats[dataset], level_2_ps[dataset], 2, 0.0
        )
    elif mode == "dense_algo":
        return lambda: lhts.reconcile_dense_algo(
            "middle_out", S_compacts[dataset], num_leaves[dataset], num_nodes[dataset], num_levels[dataset], yhats[dataset], level_2_ps[dataset], 2, 0.0
        )


d = defaultdict(lambda: defaultdict(dict))


@pytest.mark.parametrize("mode,method,dataset", itertools.product(modes, methods, DATASETS))
@pytest.mark.mpi_skip()
def test_single_process_fast(benchmark, mode, method, dataset):
    benchmark.group = method + "/" + dataset

    if method == "bottom_up":
        result = benchmark(run_bottom_up(mode, dataset))
    elif method == "middle_out":
        result = benchmark(run_middle_out(mode, dataset))
    elif method == "top_down":
        result = benchmark(run_top_down(mode, dataset))

    d[dataset][method][mode] = result
    for (i, j) in itertools.combinations(d[dataset][method].values(), 2):
        assert np.allclose(i, j, rtol=1e-3, atol=1e-5)
