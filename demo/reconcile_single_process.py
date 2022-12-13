import numpy as np
import lhts
from timeit import default_timer as timer

def main():
    
    DATA_ROOT = "m5_hobbies"
    ROOT = "/home/peiyuan20013/large-scale-hts-reconciliation/large_scale_hts_reconciliation"
    data_dir = ROOT + "/notebooks/"

    S_compact = np.load(open(data_dir + DATA_ROOT + '/m5_hierarchy_parent.npy', 'rb'))
    yhat = np.load(open(data_dir + DATA_ROOT + '/pred_tensor.npy', 'rb'))
    gt = np.load(open(data_dir + DATA_ROOT + '/gt_tensor.npy', 'rb'))
    top_down_p = np.load(open(data_dir + DATA_ROOT + '/top_down_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)
    level_2_p = np.load(open(data_dir + DATA_ROOT + '/level_2_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)
    
    print("Before Reconciliation: ", lhts.smape(yhat, gt))

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


    start = timer()
    rec = lhts.reconcile_sparse_algo("bottom_up", S_compact, 5650, 6218, 4, yhat, top_down_p, -1, 0.0)
    end = timer()
    elapsed = round(end - start, 4)
    print("Bottom up (algo): ", str(elapsed), " ", lhts.smape(rec, gt))

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
    rec = lhts.reconcile_sparse_algo("OLS", S_compact, 5650, 6218, 4, yhat, top_down_p, 0, 0.0)
    end = timer()
    elapsed = round(end - start, 4)
    print("OLS: ", str(elapsed), " ", lhts.smape(rec, gt))


    start = timer()
    rec = lhts.reconcile_sparse_algo("WLS", S_compact, 5650, 6218, 4, yhat, top_down_p, 0, 0.5)
    end = timer()
    elapsed = round(end - start, 4)
    print("WLS: ", str(elapsed), " ", lhts.smape(rec, gt))
    

if __name__ == "__main__":
    main()