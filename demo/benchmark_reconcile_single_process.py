import numpy as np
import lhts
from timeit import default_timer as timer

def main():
    ROOT = "/data/cmu/large-scale-hts-reconciliation/"
    data_dir = ROOT + "notebooks/"

    S_compact = np.load(open(data_dir + 'm5_hierarchy_parent.npy', 'rb'))
    yhat = np.load(open(data_dir + 'm5_prediction_raw/pred_tensor.npy', 'rb'))
    gt = np.load(open(data_dir + 'm5_prediction_raw/gt_tensor.npy', 'rb'))
    top_down_p = np.load(open(data_dir + 'm5_prediction_raw/top_down_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)
    level_2_p = np.load(open(data_dir + 'm5_prediction_raw/level_2_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)
    print(yhat.shape)

    print(yhat)
    print("Before Reconciliation: ", lhts.smape(yhat, gt))

    start = timer()
    rec = lhts.reconcile_sparse_algo("bottom_up", S_compact, 5650, 6218, 4, yhat, top_down_p, -1, 0.0)
    end = timer()
    elapsed = round(end - start, 4)
    print("Bottom up (sparse algo): ", str(elapsed), " ", lhts.smape(rec, gt))
    #print(rec)
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
    print("Top down (sparse algo): ", str(elapsed), " ", lhts.smape(rec, gt))
    print(rec)

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
    print("Middle out (sparse algo): ", str(elapsed), " ", lhts.smape(rec, gt))
    print(rec)

    start = timer()
    rec2 = lhts.reconcile_sparse_matrix("middle_out", S_compact, 5650, 6218, 4, yhat, level_2_p, 2, 0.0)
    end = timer()
    elapsed = round(end - start, 4)
    print("Middle out (sparse matrix): ", str(elapsed), " ", lhts.smape(rec2, gt))
    print(np.abs(rec - rec2).sum())

    start = timer()
    rec = lhts.reconcile_sparse_algo("OLS", S_compact, 5650, 6218, 4, yhat, top_down_p, 0, 0.0)
    end = timer()
    elapsed = round(end - start, 4)
    print("OLS: ", str(elapsed), " ", lhts.smape(rec, gt))
    print(rec)

    start = timer()
    rec = lhts.reconcile_sparse_algo("WLS", S_compact, 5650, 6218, 4, yhat, top_down_p, 0, 0.5)
    end = timer()
    elapsed = round(end - start, 4)
    print("WLS: ", str(elapsed), " ", lhts.smape(rec, gt))
    print(rec)

    

if __name__ == "__main__":
    main()