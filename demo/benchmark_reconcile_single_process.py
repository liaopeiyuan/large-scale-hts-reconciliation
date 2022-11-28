import numpy as np
import lhts
from timeit import default_timer as timer

def main():
    ROOT = "/data/cmu/large-scale-hts-reconciliation/"
    data_dir = ROOT + "notebooks/"

    S_compact = np.load(open(data_dir + 'm5_hierarchy_parent.npy', 'rb'))
    y_hat = np.load(open(data_dir + 'm5_prediction_raw/pred_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)
    top_down_p = np.load(open(data_dir + 'm5_prediction_raw/top_down_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)
    print(y_hat.shape)

    start = timer()
    print(lhts.reconcile("bottom_up", S_compact, top_down_p, y_hat, -1, 0.0, 5650, 6218, 4).shape)
    end = timer()
    elapsed = round(end - start, 4)
    print("Bottom up: " + str(elapsed))

    start = timer()
    print(lhts.reconcile("top_down", S_compact, top_down_p, y_hat, -1, 0.0, 5650, 6218, 4).shape)
    end = timer()
    elapsed = round(end - start, 4)
    print("Top down: " + str(elapsed))

    lhts.reconcile("blah", S_compact, top_down_p, y_hat, -1, 0.0, 5650, 6218, 4)

if __name__ == "__main__":
    main()