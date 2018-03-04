#%%

import itertools
import pandas as pd
import matplotlib.pyplot as plt
import cloudpickle as cpkl
from pathlib import Path
import numpy as np
import datetime
from functools import reduce
from fbprophet import Prophet
#%% 
#df_train = pd.read_csv(
#    'input/train.csv', usecols=[1, 2, 3, 4, 5], dtype={'onpromotion': str},
#    converters={'unit_sales': lambda u: float(u) if float(u) > 0 else 0},
#    skiprows=range(1, 124035460)
#)
train_pkl_file = 'input/train_pkl'
test_pkl_file = 'input/test_pkl'
if not Path(train_pkl_file).is_file():
    train = pd.read_csv("input/train.csv")
    print(train.head())
    transactions = pd.read_csv("input/transactions.csv")
    stores = pd.read_csv("input/stores.csv")
    oil = pd.read_csv("input/oil.csv")
    items = pd.read_csv("input/items.csv")
    holidays_events = pd.read_csv("input/holidays_events.csv")
    train_dumps = cpkl.dumps([train,transactions,stores,oil,items,holidays_events])
    with open(train_pkl_file,'wb') as f:
        f.write(train_dumps)
else:
    print('Loading train dump')
    train,transactions,stores,oil,items,holidays_events = cpkl.loads(open(train_pkl_file,'rb').read())
if not Path(test_pkl_file):
    test = pd.read_csv("input/test.csv")
    test_dumps = cpkl.dumps(test)
    with open(test_pkl_file,'wb') as f:
        f.write(test_dumps)
else:
    print("Loading test dump")
    test = cpkl.loads(open(test_pkl_file,'rb').read())
# log transform
#train["unit_sales"] = train["unit_sales"].apply(np.log1p)
# Fill NAs
train.loc[:, "unit_sales"].fillna(0, inplace=True)
# Assume missing entris imply no promotion
train.loc[:, "onpromotion"].fillna("False", inplace=True)

#%%
stores_by_transactions = transactions.groupby('store_nbr',as_index=False)['transactions'].sum()
#choose store_nbrs 42,54 and 45 for training
training_store_nbrs = [42,54,45]
train_samp = train.loc[train['store_nbr'].isin(training_store_nbrs)]
train_samp.loc[:,"date"] = pd.to_datetime(train_samp.date)
train_samp = train_samp.merge(items,on='item_nbr')
trans_samp = transactions.loc[transactions['store_nbr'].isin(training_store_nbrs)]
trans_samp.loc[:,"date"] = pd.to_datetime(trans_samp.date)
trans_samp = trans_samp.merge(stores,on='store_nbr')
trans_samp = trans_samp.merge(holidays_events,on='date',how="left")

#%%
#subset most sold item in the store with most sales for testing forecast methods
perStorePerItem=True
logTransform = False
date_range_fit = datetime.datetime(2016,8,16)
date_range_forecast = datetime.datetime(2016,9,1)
if perStorePerItem:
    storex = 45
    itemx = 414750
    trainx = train_samp[train_samp.store_nbr==45]
    trainx_item = train_samp.loc[(train_samp.store_nbr==storex) & (train_samp.item_nbr==itemx)]
    #trainx_item = trainx_item.groupby(['item_nbr','date'],as_index=False)['unit_sales'].sum()
    trainx_itemlg = trainx_item.copy()
    if logTransform:
        trainx_itemlg.unit_sales = trainx_itemlg.unit_sales.apply(np.log1p)       
    trainx_item_forfit = trainx_itemlg.loc[trainx_itemlg.date<date_range_fit]
    trainx_item_forforecast =  trainx_itemlg.loc[(trainx_itemlg.date>=date_range_fit )& (trainx_itemlg.date<date_range_forecast)]
    overallmean = np.log1p(np.mean(trainx_item.unit_sales))
    overallmeanb4fit = np.log1p(np.mean(trainx_item.loc[(trainx_item.date<date_range_fit),"unit_sales"]))
    mean_over_forecast_period = np.log1p(np.mean(trainx_item.loc[(trainx_item.date>=date_range_fit) &
                                                                 (trainx_item.date<date_range_forecast),"unit_sales"]))
   
#else:
    
#%%
#testig prophet from facebook
WEIGHTS = 1. if items.loc[items.item_nbr==itemx,'perishable'].values == 0 else 1.25
def NWRMSLE(y, pred, wts,lt=False):
    y = y.clip(0, y.max())
    pred = pred.clip(0, pred.max())
    wts = np.full_like(y,wts)
    #score = np.nansum(wts * ((np.log1p(pred) - np.log1p(y)) ** 2)) / wts.sum()
    if lt:
        score = np.nansum(wts * (np.subtract((pred), (y)) ** 2)) / wts.sum()
    else:
        score = np.nansum(wts * (np.subtract(np.log1p(pred), np.log1p(y)) ** 2)) / wts.sum()
    return np.sqrt(score)
m = Prophet()
trainx_item_forfit_forprophet = trainx_item_forfit[['date','unit_sales']]
trainx_item_forfit_forprophet= trainx_item_forfit_forprophet.rename(columns={'date':'ds','unit_sales':'y'}) 
m.fit(trainx_item_forfit_forprophet)
future = m.make_future_dataframe(periods=16)
forecast = m.predict(future)
#%%
#plot
#m.plot(forecast)
#m.plot_components(forecast)
plt.scatter(trainx_item_forforecast.date.astype(np.int64),
                             trainx_item_forforecast.unit_sales,c="red")
futureforecast = forecast.loc[forecast.ds>=date_range_fit]
plt.scatter(futureforecast.ds.astype(np.int64),futureforecast.yhat)
print("forecast {0:.4f}, overallmean {1:.4f}, overallmean_forfit{2:0.4f},meanoverforecast {3:.4f}".format(
      NWRMSLE(trainx_item_forforecast.unit_sales,futureforecast.yhat,WEIGHTS,logTransform),
      NWRMSLE(trainx_item_forforecast.unit_sales,overallmean,WEIGHTS,logTransform),
      NWRMSLE(trainx_item_forforecast.unit_sales,overallmeanb4fit,WEIGHTS,logTransform),
      NWRMSLE(trainx_item_forforecast.unit_sales,mean_over_forecast_period,WEIGHTS,logTransform)))
