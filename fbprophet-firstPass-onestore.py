#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:39:42 2017

@author: suresh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 19:31:35 2017
### for forecasting....
# create a forecast model for store.store_nbr x items.family combinations
# calculate the item_nbr-family-store_nbr fraction per store
### for prediction....
# predict the item.family forecast for the given date/store_nbr/item_nbr(family)
# scale the item.family prediction by the fraction
@author: suresh
"""

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
import calendar
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
    train.loc[:,'date']=pd.to_datetime(train.date)
    train.loc[:, "onpromotion"].fillna(False, inplace=True)
    train.loc[train.unit_sales<0,"unit_sales"] = 0
    transactions.loc[:,'date']=pd.to_datetime(transactions.date)
    oil.loc[:,'date']=pd.to_datetime(oil.date)
    train_dumps = cpkl.dumps([train,transactions,stores,oil,items,holidays_events])
    with open(train_pkl_file,'wb') as f:
        f.write(train_dumps)
else:
    print('Loading train dump')
    train,transactions,stores,oil,items,holidays_events = cpkl.loads(open(train_pkl_file,'rb').read())
if not Path(test_pkl_file).is_file():
    test = pd.read_csv("input/test.csv")
    test.loc[:,'date']=pd.to_datetime(test.date)

    test_dumps = cpkl.dumps(test)
    with open(test_pkl_file,'wb') as f:
        f.write(test_dumps)
else:
    print("Loading test dump")
    test = cpkl.loads(open(test_pkl_file,'rb').read())
# log transform
#train["unit_sales"] = train["unit_sales"].apply(np.log1p)
# Fill NAs
#train.loc[:, "unit_sales"].fillna(0, inplace=True)
# Assume missing entris imply no promotion
#train.loc[:, "onpromotion"].fillna(False, inplace=True)
#%%
def NWRMSLE(y, pred, wts,lt=False):
    y = y.clip(0, y.max())
    pred = pred.clip(0, pred.max())
    #wts = np.full_like(y,wts)
    #score = np.nansum(wts * ((np.log1p(pred) - np.log1p(y)) ** 2)) / wts.sum()
    if lt:
        score = np.nansum(wts * (np.subtract((pred), (y)) ** 2)) / wts.sum()
    else:
        score = np.nansum(wts * (np.subtract(np.log1p(pred), np.log1p(y)) ** 2)) / wts.sum()
    return np.sqrt(score)

#%%
#means = train.groupby(
#    ['item_nbr', 'store_nbr', 'onpromotion']
#)['unit_sales'].mean().to_frame('unit_sales')
#means_score = NWRMSLE()
#%%
# sample
try_on_sample=True
if try_on_sample:
    #transaction_store=transactions.merge(stores,on='store_nbr',how='left')
    stores_sample_nbrs=[44,11,17,3,43]#stores with most transactions per storey type
    stores_sample = stores.copy()
    stores_sample = stores.loc[stores.store_nbr.isin(stores_sample_nbrs)]
    train_sample = train.loc[train.store_nbr.isin(stores_sample.store_nbr)]
    train_sample.loc[:,'date']=pd.to_datetime(train_sample.date)
oil.date = pd.to_datetime(oil.date)
transactions.date = pd.to_datetime(transactions.date)
#%%
# merge with items df
if try_on_sample:
    train_sample_item = train_sample.merge(items,on="item_nbr",how="left")
    train_sample_item = train_sample_item.merge(stores,on="store_nbr",how="left")
    train_sample_item = train_sample_item.merge(oil,on="date",how="left")

#%%
# fit prophet model
failingdata =None
def fit_predict_prophet(data,holidays,test_data):
    print("fitting group {0},{1}".format(data.store_nbr.unique(),data.family.unique()))
    # for small data just return mean
#    if(len(data)<3):
#        future = pd.DataFrame({'ds':pd.date_range(test_data.date.min(),test_data.date.max(),freq="D")})
#        future = future.reindex(columns=['trend', 'trend_lower', 'trend_upper', 'yhat_lower', 'yhat_upper',
#       'extra_regressors', 'extra_regressors_lower', 'extra_regressors_upper',
#       'onpromotion', 'onpromotion_lower', 'onpromotion_upper', 'salary_day',
#       'salary_day_lower', 'salary_day_upper', 'seasonal', 'seasonal_lower',
#       'seasonal_upper', 'seasonalities', 'seasonalities_lower',
#       'seasonalities_upper', 'weekly', 'weekly_lower', 'weekly_upper',
#       'yearly', 'yearly_lower', 'yearly_upper', 'yhat'])
#        future.yhat = np.mean(data.unit_sales)
#        return future
    m = Prophet(holidays=holidays)
    #m = Prophet()
    #regressors = ['salary_day']#salary_day
    m.add_regressor('salary_day')
    m.add_regressor('onpromotion')
    m.add_regressor('dcoilwtico')
    cols = ['date','date_family_store_unit_sales','salary_day','onpromotion','dcoilwtico']#+regressors
    trainx_item_forfit_forprophet = data[cols]
    trainx_item_forfit_forprophet.loc[:,'onpromotion'] = trainx_item_forfit_forprophet.onpromotion.apply(lambda x : 1 if x else 0)
#    onpromotion_uniques = trainx_item_forfit_forprophet.onpromotion.unique()
#    salary_uniques = trainx_item_forfit_forprophet.salary_day.unique()
#    if (len(onpromotion_uniques)==1) & (onpromotion_uniques[0] ==0) :
#        trainx_item_forfit_forprophet.loc[:,'onpromotion'][:1] = 1 # set the first onpromotion to 1 - some bug in prophet prevents all values being 0s
#    if (len(salary_uniques)==1) & (salary_uniques[0] ==0) :
#        trainx_item_forfit_forprophet.loc[:,'salary_day'][:1] = 1 # set the first salary_day to 1 - some bug in prophet prevents all values being 0s
    trainx_item_forfit_forprophet.rename(columns={'date':'ds','date_family_store_unit_sales':'y'},inplace=True) 
    m.fit(trainx_item_forfit_forprophet)
    test_data = test_data.loc[(test_data.store_nbr.isin(data.store_nbr.unique()))&
                              (test_data.family.isin(data.family.unique())),["date","salary_day","onpromotion","dcoilwtico"]].rename(columns={"date":'ds'})
    print("length of test data ",len(test_data))
    forecast = m.predict(test_data)
    print('test length {0}, forecast length{1}'.format(len(test_data),len(forecast)))
    return forecast
def fit_prophet(data,holidays):
    print("fitting group {0},{1}".format(data.store_nbr.unique(),data.family.unique()))
    # for small data just return mean
#    if(len(data)<3):
#        future = pd.DataFrame({'ds':pd.date_range(test_data.date.min(),test_data.date.max(),freq="D")})
#        future = future.reindex(columns=['trend', 'trend_lower', 'trend_upper', 'yhat_lower', 'yhat_upper',
#       'extra_regressors', 'extra_regressors_lower', 'extra_regressors_upper',
#       'onpromotion', 'onpromotion_lower', 'onpromotion_upper', 'salary_day',
#       'salary_day_lower', 'salary_day_upper', 'seasonal', 'seasonal_lower',
#       'seasonal_upper', 'seasonalities', 'seasonalities_lower',
#       'seasonalities_upper', 'weekly', 'weekly_lower', 'weekly_upper',
#       'yearly', 'yearly_lower', 'yearly_upper', 'yhat'])
#        future.yhat = np.mean(data.unit_sales)
#        return future
    m = Prophet(holidays=holidays)
    #m = Prophet()
    #regressors = ['salary_day']#salary_day
    m.add_regressor('salary_day')
    m.add_regressor('onpromotion')
    m.add_regressor('dcoilwtico')
    cols = ['date','date_family_store_unit_sales','salary_day','onpromotion','dcoilwtico']#+regressors
    trainx_item_forfit_forprophet = data[cols]
    trainx_item_forfit_forprophet.loc[:,'onpromotion'] = trainx_item_forfit_forprophet.onpromotion.apply(lambda x : 1 if x else 0)
#    onpromotion_uniques = trainx_item_forfit_forprophet.onpromotion.unique()
#    salary_uniques = trainx_item_forfit_forprophet.salary_day.unique()
#    if (len(onpromotion_uniques)==1) & (onpromotion_uniques[0] ==0) :
#        trainx_item_forfit_forprophet.loc[:,'onpromotion'][:1] = 1 # set the first onpromotion to 1 - some bug in prophet prevents all values being 0s
#    if (len(salary_uniques)==1) & (salary_uniques[0] ==0) :
#        trainx_item_forfit_forprophet.loc[:,'salary_day'][:1] = 1 # set the first salary_day to 1 - some bug in prophet prevents all values being 0s
    trainx_item_forfit_forprophet.rename(columns={'date':'ds','date_family_store_unit_sales':'y'},inplace=True) 
    m.fit(trainx_item_forfit_forprophet)
    return m
# salary dates to add as regressor
def is_salary_day(date):
    return 1 if (date.day == 15 or calendar.monthrange(date.year,date.month)[1]==date.day) else 0
def predict(fitdf,data):
    #data = data[col_names]
    data = data.loc[(data.store_nbr.isin(fitdf.store_nbr.unique())) &
                                          (data.family.isin(fitdf.family.unique()))]
    print("{0},{1}".format(len(fitdf),len(data)))
    print(fitdf)
    data.rename(columns={'date':'ds'},inplace=True)
    return fitdf.iloc[0]['model'].predict(data)
#%%
# calculate means per store_nbr and item.item_nbr per date
#the following line is useless - does nothing, but we can build forecasts using other groupings for ensembling in future
#train_sample_item_summary = train_sample_item.groupby(["date","store_nbr","item_nbr"],as_index=False)["unit_sales"].mean()
holidays_for_prophet = holidays_events.copy()
holidays_for_prophet = holidays_for_prophet[["date","description"]]
holidays_for_prophet.rename(columns={'date':'ds','description':'holiday'},inplace=True)
holidays_for_prophet.lower_window=-1
holidays_for_prophet.upper_window=1
#%%
#
#1. Calculate sum of unit_sales for each store_nbr-family pair
#2. Calculate fraction for each item_nbr-family-store_nbr combination
#3. Forecast family for a given store_nbr-item_nbr
#4. Scale it by fraction
#
if try_on_sample:
    #train_sample_item.unit_sales = np.expm1(train_sample_item.unit_sales)
    train_sample_item['salary_day'] = train_sample_item.date.apply(is_salary_day)
    train_sample_item['dcoilwtico'].fillna(method='ffill',inplace=True)
    item_family_perishable = train_sample_item.groupby(['family']).agg({'perishable':'first'}).reset_index()  
    family_store_nbr_onpromotion = train_sample_item.groupby(['date','store_nbr','family']).agg({'onpromotion':np.max}).reset_index()
    family_store_nbr_date_sums = train_sample_item.groupby(['date','store_nbr','family']).agg({'unit_sales':np.sum,'salary_day':'first','dcoilwtico':'first'}).reset_index().rename(columns={'unit_sales':'date_family_store_unit_sales'})
    family_store_nbr_date_sums = family_store_nbr_date_sums.merge(family_store_nbr_onpromotion,on=['date','store_nbr','family'],how="left")
    item_nbr_store_nbr_sums=train_sample_item.groupby(['item_nbr','store_nbr','onpromotion']).agg({'unit_sales':np.sum,'family':'first'}).reset_index().rename(columns={'unit_sales':'item_store_unit_sales'})
    family_store_nbr_sums = train_sample_item.groupby(['family','store_nbr','onpromotion']).agg({'unit_sales':np.sum}).reset_index().rename(columns={'unit_sales':'family_store_unit_sales'})
    item_nbr_family_store_perc = item_nbr_store_nbr_sums.merge(family_store_nbr_sums,on=['store_nbr','family','onpromotion'],how="left")
    item_nbr_family_store_perc['item_family_perc'] = item_nbr_family_store_perc['item_store_unit_sales']/item_nbr_family_store_perc['family_store_unit_sales']
    date_range_fit = datetime.datetime(2016,8,16)
    date_range_forecast = datetime.datetime(2016,9,1)
    family_store_nbr_date_sums_val = family_store_nbr_date_sums.loc[(family_store_nbr_date_sums.date>=date_range_fit)&(family_store_nbr_date_sums.date<date_range_forecast)]
    family_store_nbr_date_sums_train = family_store_nbr_date_sums.loc[(family_store_nbr_date_sums.date<date_range_fit)]
    s_n = 44
    fam = ['AUTOMOTIVE','BEAUTY','BREAD/BAKERY']
    #bkup = family_store_nbr_date_sums_train.copy()
    #bkupval = family_store_nbr_date_sums_val.copy()
    # family_store_nbr_date_sums_train = bkup.copy()
    family_store_nbr_date_sums_train = family_store_nbr_date_sums_train.loc[(family_store_nbr_date_sums_train.store_nbr==s_n) &
                                                                            (family_store_nbr_date_sums_train.family.isin(fam))]
    family_store_nbr_date_sums_val = family_store_nbr_date_sums_val.loc[(family_store_nbr_date_sums_val.store_nbr==s_n) &
                                                                            (family_store_nbr_date_sums_val.family.isin(fam))]
    predictions = family_store_nbr_date_sums_train.groupby(['store_nbr','family']).apply(lambda x : fit_predict_prophet(x,holidays_for_prophet,family_store_nbr_date_sums_val)).reset_index()
    #fits = family_store_nbr_date_sums_train.groupby(['store_nbr','family']).apply(lambda x : fit_prophet(x,holidays_for_prophet)).reset_index()
    #fits.rename(columns={'store_nbr':'store_nbr','family':'family',0:'model'},inplace=True)
    items_for_predictions = train_sample_item.loc[(train_sample_item.store_nbr==s_n) &
                                                  (train_sample_item.family.isin(fam))&
                                                  (train_sample_item.date>=date_range_fit) &
                                                  (train_sample_item.date<date_range_forecast)]
    #family_store_for_predictions = items_for_predictions[["date","store_nbr","onpromotion","family","perishable","salary_day","dcoilwtico"]].drop_duplicates()
#    predictions = pd.DataFrame()
#    for name, df in fits.groupby(['store_nbr','family']):
#        predict(df,family_store_for_predictions.loc[(family_store_for_predictions.store_nbr.isin(df.store_nbr.unique())) &
#                                                    (family_store_for_predictions.family.isin(df.family.unique()))])
    #predictions =fits.groupby(['store_nbr','family']).apply(lambda x : predict(x,family_store_for_predictions)).reset_index()
    items_for_predictions = items_for_predictions.merge(predictions,left_on=['date','store_nbr','family'],
                                                        right_on=['ds','store_nbr','family'],how="left")
    items_for_predictions.rename(columns={"onpromotion_x":"onpromotion"},inplace=True)
    items_for_predictions = items_for_predictions.merge(item_nbr_family_store_perc,on=['item_nbr','store_nbr','family','onpromotion'],how="left",copy=False)
    items_for_predictions['predicted_unit_sales'] = items_for_predictions['yhat']*items_for_predictions['item_family_perc']
    items_for_predictions['WEIGHTS'] = items_for_predictions['perishable'].apply(lambda x : 1.25 if x==1 else 1 )
    print(NWRMSLE(items_for_predictions.unit_sales,items_for_predictions.predicted_unit_sales,items_for_predictions.WEIGHTS,lt=True))
    
if False:
    ts = transactions.groupby('date').agg({'transactions':np.sum}).reset_index()
    ts.plot('date','transactions')
    ds = train.groupby('date').agg({'unit_sales':'count'}).reset_index()
    ds.plot('date','unit_sales')
    returns = train.loc[train.unit_sales<0]
    returns_sum =returns.groupby('date').agg({'unit_sales':sum}).reset_index()
    returns_sum.plot(x='date',y='unit_sales')