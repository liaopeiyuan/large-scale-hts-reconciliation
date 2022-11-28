import numpy as np
import lhts
from timeit import default_timer as timer

def main():
    ROOT = "/data/cmu/large-scale-hts-reconciliation/"
    data_dir = ROOT + "notebooks/"

    S_compact = np.load(open(data_dir + 'm5_hierarchy_parent.npy', 'rb'))

    start = timer()
    print(lhts.construct_G_bottom_up(S_compact, 30490, 33549, 4).shape)
    end = timer()
    print(end - start)

if __name__ == "__main__":
    main()