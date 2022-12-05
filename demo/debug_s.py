import numpy as np
import lhts
from timeit import default_timer as timer

def main():
    ROOT = "/data/cmu/large-scale-hts-reconciliation/"
    data_dir = ROOT + "notebooks/"

    S_compact = np.load(open(data_dir + 'm5_hierarchy_parent.npy', 'rb'))
    S = lhts.construct_S(S_compact, 5650, 6218, 4)
    St = np.trnspose(S)
    G = np.linalg.inv(St * S) * St
    print(G)
    

if __name__ == "__main__":
    main()