import lhts
import numpy as np
from timeit import default_timer as timer
import pytest

ROOT = "/data/cmu/large-scale-hts-reconciliation/"
data_dir = ROOT + "notebooks/"

S_compact = np.load(open(data_dir + 'm5_hierarchy_parent.npy', 'rb'))
yhat = np.load(open(data_dir + 'm5_prediction_raw/pred_tensor.npy', 'rb'))
gt = np.load(open(data_dir + 'm5_prediction_raw/gt_tensor.npy', 'rb'))
top_down_p = np.load(open(data_dir + 'm5_prediction_raw/top_down_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)
level_2_p = np.load(open(data_dir + 'm5_prediction_raw/level_2_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)

methods = ["bottom_up", "middle_out", "top_down"]
modes = ["dense", "sparse"]

def run_bottom_up(mode):
    if mode == "sparse":
        return lambda: lhts.reconcile_sparse_algo("bottom_up", S_compact, 5650, 6218, 4, yhat, top_down_p, -1, 0.0)
    else:
        return lambda: lhts.reconcile_dense_algo("bottom_up", S_compact, 5650, 6218, 4, yhat, top_down_p, -1, 0.0)

def run_main():

    print(yhat.shape)

    print(yhat)
    print("Before Reconciliation: ", lhts.smape(yhat, gt))

    start = timer()
    rec = lhts.reconcile_sparse_algo("bottom_up", S_compact, 5650, 6218, 4, yhat, top_down_p, -1, 0.0)
    end = timer()
    elapsed = round(end - start, 4)
    print("Bottom up (algo): ", str(elapsed), " ", lhts.smape(rec, gt))

    print(np.abs(yhat[:5650,-1:] - rec[:5650,-1:]).sum())
    print(np.abs(yhat[5650:,-1:] - rec[5650:,-1:]).sum())

    start = timer()
    rec2 = lhts.reconcile_sparse_matrix("bottom_up", S_compact, 5650, 6218, 4, yhat, top_down_p, -1, 0.0)
    end = timer()
    elapsed = round(end - start, 4)
    print("Bottom up (sparse matrix): ", str(elapsed), " ", lhts.smape(rec2, gt))
    print(np.abs(rec - rec2).sum())

    start = timer()
    rec = lhts.reconcile_sparse_algo("top_down", S_compact, 5650, 6218, 4, yhat, top_down_p, -1, 0.0)
    end = timer()
    elapsed = round(end - start, 4)
    print("Top down (algo): ", str(elapsed), " ", lhts.smape(rec, gt))


    start = timer()
    rec2 = lhts.reconcile_sparse_matrix("top_down", S_compact, 5650, 6218, 4, yhat, top_down_p, -1, 0.0)
    end = timer()
    elapsed = round(end - start, 4)
    print("Top down (sparse matrix): ", str(elapsed), " ", lhts.smape(rec2, gt))
    print(np.abs(rec - rec2).sum())

    start = timer()
    rec = lhts.reconcile_sparse_algo("middle_out", S_compact, 5650, 6218, 4, yhat, level_2_p, 2, 0.0)
    end = timer()
    elapsed = round(end - start, 4)
    print("Middle out (algo): ", str(elapsed), " ", lhts.smape(rec, gt))


    start = timer()
    rec2 = lhts.reconcile_sparse_matrix("middle_out", S_compact, 5650, 6218, 4, yhat, level_2_p, 2, 0.0)
    end = timer()
    elapsed = round(end - start, 4)
    print("Middle out (sparse matrix): ", str(elapsed), " ", lhts.smape(rec2, gt))
    print(np.abs(rec - rec2).sum())

    """
    start = timer()
    rec = lhts.reconcile_sparse_algo("OLS", S_compact, 5650, 6218, 4, yhat, top_down_p, 0, 0.0)
    end = timer()
    elapsed = round(end - start, 4)
    print("OLS: ", str(elapsed), " ", lhts.smape(rec, gt))

    start = timer()
    rec = lhts.reconcile_sparse_algo("WLS", S_compact, 5650, 6218, 4, yhat, top_down_p, 0, 0.5)
    end = timer()
    elapsed = round(end - start, 4)
    print("WLS: ", str(elapsed), " ", lhts.smape(rec, gt))
    """

    return True

@pytest.mark.parametrize(
    "mode", modes
)
def test_main(benchmark, mode):
    benchmark.group = '%s - perf' % mode
    # benchmark something
    result = benchmark(run_bottom_up(mode))

    # Extra code, to verify that the run completed correctly.
    # Sometimes you may want to check the result, fast functions
    # are no good if they return incorrect results :-)
    assert True
