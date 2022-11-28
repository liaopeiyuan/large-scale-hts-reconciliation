import numpy as np
import lhts
from timeit import default_timer as timer

SIZE = 30000

def main():
    S = np.random.normal(0, 1, size=(SIZE, SIZE)).astype(int)
    G = np.random.normal(0, 1, size=(SIZE, SIZE)).astype(int)
    yhat = np.random.normal(size=(SIZE, 1))

    start = timer()
    print(lhts.reconcile(S, G, yhat).shape)
    end = timer()
    print(end - start)

if __name__ == "__main__":
    main()