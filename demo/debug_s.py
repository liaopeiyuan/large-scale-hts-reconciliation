import numpy as np
import lhts
from timeit import default_timer as timer
from numba import jit

@jit
def f(S):
    St = np.transpose(S)
    G = np.matmul(np.linalg.inv(St @ S), St)
    return G


def main():
    ROOT = "/data/cmu/large-scale-hts-reconciliation/"
    data_dir = ROOT + "notebooks/"

    S_compact = np.load(open(data_dir + 'm5_hierarchy_parent.npy', 'rb'))
    S = lhts.build_S_sparse(S_compact, 5650, 6218, 4)
    G = f(S)
    print(G)
    

if __name__ == "__main__":
    main()