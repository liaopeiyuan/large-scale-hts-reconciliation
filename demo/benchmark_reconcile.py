import numpy as np
import lhts
from timeit import default_timer as timer


def main():
    S = np.random.normal(size=(30000, 30000)).astype(int)
    G = np.random.normal(size=(30000, 30000)).astype(int)
    yhat = np.random.normal(size=(30000, 1)).astype(int)

    start = timer()
    print(lhts.det(S, G, yhat).shape)
    end = timer()
    print(end - start)

if __name__ == "__main__":
    main()