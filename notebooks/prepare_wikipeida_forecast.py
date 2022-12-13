#!/usr/bin/env python
# coding: utf-8

# ## Generating Benchmark Data for Wikipedia Dataset
#
# The data is taken from the [Web Traffic Time Series Forecasting
# ](https://www.kaggle.com/competitions/web-traffic-time-series-forecasting/overview) challenge on Kaggle, a research competiton organized by Google ($25,000).

# In[1]:

ROOT = "/data/cmu/large-scale-hts-reconciliation/notebooks/"
data_dir = ROOT  # + "web-traffic-time-series-forecasting/"


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from tqdm import tqdm


# In[3]:


df = pd.read_csv(data_dir + "train_1.csv").fillna(0)
df


# In[13]:


import dask.dataframe as dd

ddf = dd.from_pandas(df, npartitions=120)

forecast_horizon = 100


def predict(row):
    data = pd.DataFrame(
        {
            "ds": (row.index)[1:][-2 * forecast_horizon : -forecast_horizon],
            "y": (row.values)[1:][-2 * forecast_horizon : -forecast_horizon],
        }
    )
    m = Prophet()
    m.fit(data)

    future = m.make_future_dataframe(periods=forecast_horizon)
    future.tail()
    forecast = m.predict(future)

    return row.Page, forecast[["yhat"]][-forecast_horizon:].values.reshape(-1)


# In[14]:


dask_series = ddf.apply(predict, axis=1, meta=("float", "object"))


# In[15]:


if __name__ == "__main__":
    result = dask_series.compute(scheduler="processes")

    import pickle

    pickle.dump(result, open("result.pkl", "wb"))
