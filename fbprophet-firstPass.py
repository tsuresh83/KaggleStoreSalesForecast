#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 19:31:35 2017

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
train.loc[:, "onpromotion"].fillna(False, inplace=True)
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
#train_sample = train.copy()
stores_sample = stores.copy()
if try_on_sample:
    stores_sample = stores_sample.sample(frac=0.1,replace=False)
stores_sample_nbrs = [49, 50,  7, 37, 19]
stores_sample = stores.loc[stores.store_nbr.isin(stores_sample_nbrs)]
train_sample = train.loc[train.store_nbr.isin(stores_sample.store_nbr)]
train_sample.loc[:,'date']=pd.to_datetime(train_sample.date)
oil.date = pd.to_datetime(oil.date)
transactions.date = pd.to_datetime(transactions.date)
#%%
# merge with items df
train_sample_item = train_sample.merge(items,on="item_nbr",how="left")
train_sample_item = train_sample_item.merge(stores,on="store_nbr",how="left")
train_sample_item = train_sample_item.merge(oil,on="date",how="left")

#%%
# fit prophet model
failingdata =None
def fit_predict_prophet(data,holidays,test_data):
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
#    m.add_seasonality("weekly",7,fourier_order=3)
#    m.add_seasonality("half_monthly",15,fourier_order=5)
#    m.add_seasonality("monthly",30.5,fourier_order=7)
#    m.add_seasonality("quarterly",90,fourier_order=8)
#    m.add_seasonality("yearly",365.5,fourier_order=12)
    cols = ['date','unit_sales','salary_day','onpromotion','dcoilwtico']#+regressors
    trainx_item_forfit_forprophet = data[cols]
    trainx_item_forfit_forprophet.loc[:,'onpromotion'] = trainx_item_forfit_forprophet.onpromotion.apply(lambda x : 1 if x else 0)
    onpromotion_uniques = trainx_item_forfit_forprophet.onpromotion.unique()
    salary_uniques = trainx_item_forfit_forprophet.salary_day.unique()
    if (len(onpromotion_uniques)==1) & (onpromotion_uniques[0] ==0) :
        trainx_item_forfit_forprophet.loc[:,'onpromotion'][:1] = 1 # set the first onpromotion to 1 - some bug in prophet prevents all values being 0s
    if (len(salary_uniques)==1) & (salary_uniques[0] ==0) :
        trainx_item_forfit_forprophet.loc[:,'salary_day'][:1] = 1 # set the first salary_day to 1 - some bug in prophet prevents all values being 0s
    trainx_item_forfit_forprophet.rename(columns={'date':'ds','unit_sales':'y'},inplace=True) 
    m.fit(trainx_item_forfit_forprophet)
    future = pd.DataFrame({'ds':pd.date_range(test_data.date.min(),test_data.date.max(),freq="D")})
    #future = m.make_future_dataframe(periods=future_days_to_predict)
    future['salary_day'] = future['ds'].apply(is_salary_day)
#    test_data_for_item_store = test_data.loc[(test_data.store_nbr.isin(data.store_nbr.unique())) &
#                                             (test_data.item_nbr.isin(data.item_nbr.unique()))]
    future = future.merge(test_data[["date","onpromotion"]],how="right",left_on="ds",right_on="date")[["ds","salary_day","onpromotion"]]
    future.loc[:, "onpromotion"].fillna(False, inplace=True)
    future.loc[:,'onpromotion'] = future.onpromotion.apply(lambda x : 1 if x else 0)
    future = future.merge(test_data[["date","dcoilwtico"]],how="left",left_on="ds",right_on="date")[["ds","salary_day","onpromotion","dcoilwtico"]]
    forecast = m.predict(future)
    return forecast
def fit_prophet(data,holidays):
    m = Prophet(holidays=holidays)
    #m = Prophet()
    # add additional regressors
    m.add_regressor('salary_day')
    m.add_regressor('onpromotion')
    m.add_regressor('dcoilwtico')
    cols = ['date','unit_sales','salary_day','onpromotion','dcoilwtico']#+regressors
    trainx_item_forfit_forprophet = data[cols]
    trainx_item_forfit_forprophet.loc[:,'onpromotion'] = trainx_item_forfit_forprophet.onpromotion.apply(lambda x : 1 if x else 0)
    #prophet has issues if the regressor has only zero values - this is a work around
    onpromotion_uniques = trainx_item_forfit_forprophet.onpromotion.unique()
    salary_uniques = trainx_item_forfit_forprophet.salary_day.unique()
    if (len(onpromotion_uniques)==1) & (onpromotion_uniques[0] ==0) :
        trainx_item_forfit_forprophet.loc[:,'onpromotion'][:1] = 1 # set the first onpromotion to 1 - some bug in prophet prevents all values being 0s
    if (len(salary_uniques)==1) & (salary_uniques[0] ==0) :
        trainx_item_forfit_forprophet.loc[:,'salary_day'][:1] = 1 # set the first salary_day to 1 - some bug in prophet prevents all values being 0s
    # prophet bug work around ends here
    trainx_item_forfit_forprophet.rename(columns={'date':'ds','unit_sales':'y'},inplace=True) 
    m.fit(trainx_item_forfit_forprophet)
    return m
# salary dates to add as regressor
def is_salary_day(date):
    return 1 if (date.day == 15 or calendar.monthrange(date.year,date.month)[1]==date.day) else 0
#%%
# calculate means per store_nbr and item.item_nbr per date
#the following line is useless - does nothing, but we can build forecasts using other groupings for ensembling in future
#train_sample_item_summary = train_sample_item.groupby(["date","store_nbr","item_nbr"],as_index=False)["unit_sales"].mean()
holidays_for_prophet = holidays_events.copy()
holidays_for_prophet = holidays_for_prophet[["date","description"]]
holidays_for_prophet.rename(columns={'date':'ds','description':'holiday'},inplace=True)
holidays_for_prophet.lower_window=1
holidays_for_prophet.upper_window=1
#%%
# train on sample data
# test data has new store-item pairs - soln : forecast using store.cluster, store.state, store.type and items.item_nbr, items.family,items.class pairs
# even in the event of no missing data all these combinations should be used for ensembling
train_sample_item['salary_day'] = train_sample_item.date.apply(is_salary_day)
#train_sample_item['month'] = train_sample_item.date.apply(lambda x : x.month)
#train_sample_item['weekofmonth'] = train_sample_item.date.apply(lambda x : (x.day-1)//7 +1)
test_sample=True
calculate_means=False
if test_sample:
    date_range_fit = datetime.datetime(2016,8,16)
    date_range_forecast = datetime.datetime(2016,9,1)
    test_items = [1412379,  108696, 1005464, 1945572]
    #0.45 and 0.41 scores with prophet default and mean
    #0.46 and 0.41 with prophet holidays and mean
    #0.46 and 0.41 with prophet holidays+salary and mean
    #0.46 and 0.41 with prophet holidays+onpromotion and mean
    #0.46 and 0.41 with prophet holidays+onpromotion+salary_day and mean
    actual_values = train_sample_item.loc[(train_sample_item.item_nbr.isin(test_items))&(train_sample_item.date>=date_range_fit)&(train_sample_item.date<date_range_forecast)]
    train_sample_item_forecast = train_sample_item.loc[(train_sample_item.item_nbr.isin(test_items))&(train_sample_item.date<date_range_fit)]
    ###pe-per item fit - most expensive
    #forecast_values = train_sample_item.loc[(train_sample_item.item_nbr.isin(test_items))&(train_sample_item.date<date_range_fit)].groupby(['store_nbr','item_nbr']).apply(lambda x: fit_prophet(x,holidays_for_prophet,actual_values)).reset_index()
    
#    forecast_actual = actual_values.merge(forecast_values,left_on=['date','store_nbr','item_nbr'],
#                                          right_on=['ds','store_nbr','item_nbr'],how="left")
    #forecast_actual = forecast_actual[['date','store_nbr','item_nbr','perishable','yhat','unit_sales']]
    #forecast_actual['WEIGHTS'] = forecast_actual.perishable.apply(lambda x : 1. if x==0 else 1.25)
    #print(NWRMSLE(forecast_actual.unit_sales,forecast_actual.yhat,forecast_actual.WEIGHTS))
    ##now every store-item attribute combination
    store_cols = stores.columns
    item_cols = items.columns
    store_item_combn = [[x,y] for x in store_cols for y in item_cols ]
    combn_forecasts ={}
    for combn in store_item_combn:
        if combn[1] =='item_nbr':
            continue
        print("fitting .... ",combn)
        combn_mean = train_sample_item_forecast.groupby(['date']+combn+['onpromotion'])['unit_sales','salary_day','dcoilwtico'].mean().reset_index()
        combn_mean['dcoilwtico'].fillna(method='ffill',inplace=True)
        #combn_mean_for_prediction = actual_values.groupby(['date']+combn+['onpromotion'])['unit_sales','salary_day','dcoilwtico'].mean().reset_index()
        #combn_mean_for_prediction['dcoilwtico'].fillna(method='ffill',inplace=True)
        data_for_prediciton = actual_values[['date']+combn+['onpromotion','salary_day','dcoilwtico']].drop_duplicates()
        combn_forecasts['-'.join(str(t) for t in combn)] = combn_mean.groupby(combn).apply(lambda x: fit_prophet(x,holidays_for_prophet)).reset_index()
    mean_store_item = train_sample_item_forecast.groupby(['store_nbr','item_nbr'])['unit_sales'].mean().reset_index()
    mean_store_item.rename(columns={"unit_sales":"mean"},inplace=True)
    mean_actual = actual_values.merge(mean_store_item,left_on=['store_nbr','item_nbr'],
                                          right_on=['store_nbr','item_nbr'],how="left")
    mean_actual = mean_actual[['date','store_nbr','item_nbr','perishable','mean','unit_sales']]
    mean_actual['WEIGHTS'] = mean_actual.perishable.apply(lambda x : 1. if x==0 else 1.25)
    print(NWRMSLE(mean_actual.unit_sales,mean_actual['mean'],mean_actual.WEIGHTS))
    ### this section is for calculating means per store attrib, item attrib combination -
    ### should be explored later on in the competition 
    ### can be used as regressors in an ML model with other forecast values
    if calculate_means:
        store_item_promo_combn_means = [[x,y,z] for x in store_cols for y in item_cols for z in ['onpromotion'] ]
        combn_means ={}
        for combn in store_item_promo_combn_means:
            print("averaging .... ",combn)
            colname = '-'.join(combn)+"_wom_month_mean"
            #store_item_means[colname] = train_sample_item_forecast.groupby(combn+['weekofmonth','month'])['unit_sales'].transform(np.mean)
            tmp = train_sample_item_forecast.groupby(combn+['weekofmonth','month'])['unit_sales'].mean().reset_index()
            tmp.rename(columns={'unit_sales':colname},inplace=True)
            combn_means['-'.join(combn)] = tmp
        store_item_means = stores_sample.copy()
        store_item_means["onpromotion"]=False
        store_item_means2 = store_item_means.copy()
        store_item_means2["onpromotion"]=True
        store_item_means = pd.concat([store_item_means,store_item_means2])
        store_nbr_ctr =0
        item_nbr_ctr=0
        item_store_means=items.loc[items.item_nbr.isin(train_sample_item_forecast.item_nbr.unique())]
        item_store_means['onpromotion']=False
        item_store_means2 = item_store_means.copy()
        item_store_means2['onpromotion']=True
        item_store_means = pd.concat([item_store_means,item_store_means2])
        for k,df in combn_means.items():
            gp = k.split('-')
            if gp[1] == 'item_nbr':
                item_nbr_ctr+=1
                if item_nbr_ctr==1:
                    item_store_means = item_store_means.merge(df,on=['item_nbr','onpromotion'])
                    print("store_nbr_ctr ",store_nbr_ctr)
                    print(item_store_means.columns)
                else:
                    item_store_means = item_store_means.merge(df,on=['item_nbr','onpromotion','weekofmonth','month'],how="left")
    
            if gp[0] == 'store_nbr':
                store_nbr_ctr+=1
                print(store_item_means.columns)
                print(df.columns)
                print("---------------")
                if store_nbr_ctr==1:
                    store_item_means = store_item_means.merge(df,on=['store_nbr','onpromotion'])
                    print("store_nbr_ctr ",store_nbr_ctr)
                    print(store_item_means.columns)
                else:
                    store_item_means = store_item_means.merge(df,on=['store_nbr','onpromotion','weekofmonth','month'],how="left")
            else:
                store_item_means = store_item_means.merge(df, on = gp+['weekofmonth','month'],how="left")
        

#%%
# do on the entire dataset
train_item = train.merge(items,on="item_nbr",how="left")
train_item.loc[:,'date']=pd.to_datetime(train_item.date)
train_item['salary_day'] = train_item.date.apply(is_salary_day)
#%% fit prophet and predict
forecast_values = train_item.groupby(['store_nbr','item_nbr']).apply(lambda x: fit_prophet(x,holidays_for_prophet,test)).reset_index()
