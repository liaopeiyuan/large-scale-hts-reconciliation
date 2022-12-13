#!/usr/bin/env python3

import mpi4py

mpi4py.rc.threaded = True
mpi4py.rc.thread_level = "funneled"
from mpi4py import MPI
from lhts import Distributed
import lhts
from prophet import Prophet

import numpy as np
import pandas as pd
from timeit import default_timer as timer
from tqdm import tqdm 

def main():
    METHOD = "bottom_up"
    TIME_HORIZON = 28

    DATA_ROOT = "m5_hobbies"
    ROOT = "/home/peiyuan20013/large-scale-hts-reconciliation/large_scale_hts_reconciliation"
    data_dir = ROOT + "/notebooks/"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    distrib = Distributed()

    start = timer()

    S_compact = np.load(open(data_dir + DATA_ROOT + '/m5_hierarchy_parent.npy', 'rb'))
    
    if (METHOD == "middle_out"):
        P = np.load(open(data_dir + DATA_ROOT + '/top_down_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)
    else:
        P = np.load(open(data_dir + DATA_ROOT + '/level_2_tensor.npy', 'rb'))[:, 0].reshape(-1, 1)

    df = pd.read_csv(data_dir + DATA_ROOT + '/m5_historical.csv')
    total_len = len(df)
    len_per_rank = (total_len + size - 1) // size
    
    df_slice = df.iloc[len_per_rank * rank : min(total_len, len_per_rank * (rank + 1)), :]

    yhat = np.zeros((len(df_slice), TIME_HORIZON))
    gt = np.zeros((len(df_slice), TIME_HORIZON))

    if rank == 0:
        it = tqdm(df_slice.iterrows())
    else:
        it = df_slice.iterrows()

    for i, row in it:
        data = pd.DataFrame({'ds': (row.index)[1:-4][-TIME_HORIZON:], 'y':(row.values)[1:-4][-TIME_HORIZON:]})
        m = Prophet()
        m.fit(data)

        future = m.make_future_dataframe(periods=TIME_HORIZON)
        forecast = m.predict(future)

        yhat[i, :] = forecast[['yhat']][-TIME_HORIZON:].values.reshape(-1)
        gt[i, :] = (row.values)[1:-4][-TIME_HORIZON:]

    start = timer()
    rec = distrib.reconcile_dp_optimized(METHOD, S_compact, 5650, 6218, 4, yhat, P, 2, 1.5)
    end = timer()
    elapsed = round(end - start, 4)
    if (rank == size - 1):
        print("Reconciliation with " + METHOD + " done: ", str(elapsed), " ", lhts.smape(rec, gt))


if __name__ == "__main__":
    main()