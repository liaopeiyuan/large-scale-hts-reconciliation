#!/usr/bin/env python
# coding: utf-8

# ## Generating Benchmark Data for Wikipedia Dataset
# 
# The data is taken from the [Web Traffic Time Series Forecasting
# ](https://www.kaggle.com/competitions/web-traffic-time-series-forecasting/overview) challenge on Kaggle, a research competiton organized by Google ($25,000).

# In[1]:


ROOT = "/Users/liaopeiyuan/Documents/"
data_dir = ROOT + "web-traffic-time-series-forecasting/"


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from tqdm import tqdm


# In[3]:


df = pd.read_csv(data_dir + 'train_1.csv').fillna(0)
df


# In[13]:


import dask.dataframe as dd

ddf = dd.from_pandas(df, npartitions=10000)

forecast_horizon = 100

def predict(row):
    data = pd.DataFrame({'ds': (row.index)[1:][-forecast_horizon:], 'y':(row.values)[1:][-forecast_horizon:]})
    m = Prophet()
    m.fit(data)
    
    future = m.make_future_dataframe(periods=forecast_horizon)
    future.tail()
    forecast = m.predict(future)

    return row.Page, forecast[['yhat']][-forecast_horizon:].values.reshape(-1)


# In[14]:


dask_series = ddf.apply(predict, axis=1, meta=('float', 'object'))  


# In[15]:


result = dask_series.compute()


# In[20]:


import pickle
pickle.dump(result[0], open('result.pkl', 'wb'))


# In[ ]:


forecast_horizon = 100
node_list = []
forecasts = np.zeros((len(df), forecast_horizon)).astype(float)


# In[ ]:


for i, ro in tqdm(df.iterrows(), total=len(df)):
    node_list.append(ro.Page)
    data = pd.DataFrame({'ds': (ro.index)[1:], 'y':(ro.values)[1:]})
    m = Prophet()
    m.fit(data)
    
    future = m.make_future_dataframe(periods=forecast_horizon)
    future.tail()
    forecast = m.predict(future)

    forecasts[i,:] = forecast[['yhat']][-forecast_horizon:].values.reshape(-1)


# In[ ]:




