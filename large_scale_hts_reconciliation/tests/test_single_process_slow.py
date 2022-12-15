import lhts
import numpy as np
import pytest
import itertools
from collections import defaultdict

DATASETS = ["tourism_small", "labour"]

hierarchy_prefix = {
    "m5_hobbies": "m5",
    "m5_full": "m5",
    "wikipedia": "wikipedia",
    "tourism_small": "tourism",
    "labour": "labour",
}
num_leaves = {
    "m5_hobbies": 5650,
    "m5_full": 30490,
    "wikipedia": 145063,
    "tourism_small": 56,
    "labour": 32,
}
num_nodes = {
    "m5_hobbies": 6218,
    "m5_full": 33549,
    "wikipedia": 308004,
    "tourism_small": 89,
    "labour": 57,
}
num_levels = {
    "m5_hobbies": 4,
    "m5_full": 4,
    "wikipedia": 4,
    "tourism_small": 4,
    "labour": 4,
}


ROOT = "/home/peiyuan20013/large-scale-hts-reconciliation/large_scale_hts_reconciliation"  # "/data/cmu/large-scale-hts-reconciliation/"
data_dir = ROOT + "/notebooks/"

S_compacts = {}
top_down_ps = {}
level_2_ps = {}
gts = {}
yhats = {}

for DATA_ROOT in DATASETS:
    S_compact = np.load(
        open(
            data_dir
            + DATA_ROOT
            + "/"
            + hierarchy_prefix[DATA_ROOT]
            + "_hierarchy_parent.npy",
            "rb",
        )
    )
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

methods = ["OLS", "WLS"]
modes = ["dense_algo", "sparse_algo", "dense_matrix", "sparse_matrix"]


def run(mode, method, dataset):
    if mode == "sparse_matrix":
        return lambda: lhts.reconcile_sparse_matrix(
            method,
            S_compacts[dataset],
            num_leaves[dataset],
            num_nodes[dataset],
            num_levels[dataset],
            yhats[dataset],
            top_down_ps[dataset],
            -1,
            1.5,
        )
    elif mode == "sparse_algo":
        return lambda: lhts.reconcile_sparse_algo(
            method,
            S_compacts[dataset],
            num_leaves[dataset],
            num_nodes[dataset],
            num_levels[dataset],
            yhats[dataset],
            top_down_ps[dataset],
            -1,
            1.5,
        )
    elif mode == "dense_matrix":
        return lambda: lhts.reconcile_dense_matrix(
            method,
            S_compacts[dataset],
            num_leaves[dataset],
            num_nodes[dataset],
            num_levels[dataset],
            yhats[dataset],
            top_down_ps[dataset],
            -1,
            1.5,
        )
    elif mode == "dense_algo":
        return lambda: lhts.reconcile_dense_algo(
            method,
            S_compacts[dataset],
            num_leaves[dataset],
            num_nodes[dataset],
            num_levels[dataset],
            yhats[dataset],
            top_down_ps[dataset],
            -1,
            1.5,
        )


d = defaultdict(lambda: defaultdict(dict))


@pytest.mark.parametrize(
    "mode,method,dataset", itertools.product(modes, methods, DATASETS)
)
@pytest.mark.benchmark(
    min_rounds=1,
    max_time=10,
)
@pytest.mark.mpi_skip()
def test_single_process_slow(benchmark, mode, method, dataset):
    
    benchmark.group = method + "/" + dataset

    result = benchmark.pedantic(run(mode, method, dataset), iterations=1, rounds=20)

    # Due to instability of QR factorization, we omit the check
    # d[dataset][method][mode] = result
    # for (i, j) in itertools.combinations(d[dataset][method].values(), 2):
    #   assert np.allclose(i, j, rtol=1e-3, atol=1e-5)