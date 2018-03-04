#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:24:46 2017

@author: suresh

### for forecasting....
# create a forecast model for store.store_nbr x items.item_nbr combinations
### for prediction....
# predict the store_nbr-item_nbr forecast for the given date/store_nbr/item_nbr
"""

#%%

import random
from itertools import product
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
    
#%%
# Consider data only after 2015 - the oil price is relatively stable after 2015
big_bang = datetime.datetime(2014,12,31)
train = train.loc[train.date>big_bang]
#train.loc[:, "onpromotion"].fillna(False, inplace=True)
transactions = transactions.loc[transactions.date>big_bang]
#oil = oil.loc[oil.date>big_bang]
#%%
# prepare holidays df for prophet
holidays_for_prophet = holidays_events.copy()
holidays_for_prophet = holidays_for_prophet[holidays_for_prophet.transferred==False]
holidays_for_prophet = holidays_for_prophet[["date","description"]]
holidays_for_prophet.rename(columns={'date':'ds','description':'holiday'},inplace=True)
holidays_for_prophet.lower_window=1
holidays_for_prophet.upper_window=-2
oil['dcoilwtico'].fillna(method='ffill',inplace=True)
#%%
# define functions and other variables for fitting and prediction
prediction_window_start_date = datetime.date(2017,8,16)
prediction_window_end_date = datetime.date(2017,8,31)
def fit_prophet(data,holidays):
    if(data.shape[0]<3):
        return np.nan
    m = Prophet(holidays=holidays)
    #m = Prophet()
    # add additional regressors
    m.add_regressor('salary_day')
    m.add_regressor('onpromotion')
    m.add_regressor('dcoilwtico')
    cols = ['date','unit_sales','salary_day','onpromotion','dcoilwtico']#+regressors
    trainx_item_forfit_forprophet = data[cols]
    #trainx_item_forfit_forprophet.loc[:,'onpromotion'] = trainx_item_forfit_forprophet.onpromotion.apply(lambda x : 1 if x else 0)
    trainx_item_forfit_forprophet.rename(columns={'date':'ds','unit_sales':'y'},inplace=True) 
    #prophet has issues if the regressor has only zero values - this is a work around
    onpromotion_uniques = trainx_item_forfit_forprophet.onpromotion.unique()
    salary_uniques = trainx_item_forfit_forprophet.salary_day.unique()
    if (len(onpromotion_uniques)==1) & (onpromotion_uniques[0] ==0) :
        trainx_item_forfit_forprophet.loc[:,'onpromotion'][:1] = 1 # set the first onpromotion to 1 - some bug in prophet prevents all values being 0s
    if (len(salary_uniques)==1) & (salary_uniques[0] ==0) :
        trainx_item_forfit_forprophet.loc[:,'salary_day'][:1] = 1 # set the first salary_day to 1 - some bug in prophet prevents all values being 0s
    # prophet bug work around ends here
    m.fit(trainx_item_forfit_forprophet)
    return m
# salary dates to add as regressor
def is_salary_day(date):
    return 1 if (date.day == 15 or calendar.monthrange(date.year,date.month)[1]==date.day) else 0
def predictWithData(data,fitdf):
    #data = data[col_names]
    modelRow = fitdf[(fitdf.store_nbr.isin(data.store_nbr.unique())) &
                     (fitdf.item_nbr.isin(data.item_nbr.unique()))]
    data.rename(columns={"date":'ds'},inplace=True)
    # if store_nbr-item_nbr combination not in train data return nan
    if len(modelRow)==0:
        print("{0},{1}".format(data.store_nbr.unique(),data.item_nbr.unique()))
        future = pd.DataFrame({'ds':data.ds})
        future['yhat_lower'] = np.nan
        future['yhat_upper'] = np.nan
        future['yhat'] = np.nan
        future['store_nbr'] = data.store_nbr
        future['item_nbr'] = data.item_nbr
        return future
    
    print("{0},{1}".format(len(modelRow),len(data)))
    print(modelRow)
    #data.rename(columns={'date':'ds'},inplace=True)
    preds= modelRow.iloc[0]['Model'].predict(data)[["ds","yhat_lower","yhat_upper","yhat"]]
    preds['store_nbr'] = data.store_nbr
    preds['item_nbr'] = data.item_nbr
    return preds
def predict(fitdf,mindate,maxdate,oil):
    #data = data[col_names]
    data = create_prophet_futuredf(mindate,maxdate,oil)
    print("{0},{1}".format(len(fitdf),len(data)))
    print(fitdf)
    data.rename(columns={'date':'ds'},inplace=True)
    return fitdf.iloc[0]['Model'].predict(data)
def create_prophet_futuredf(mindate,maxdate,oil):
    future_df_for_prophet = pd.DataFrame({'ds':pd.date_range(mindate,maxdate,freq="D")})
    future_df_for_prophet['salary_day'] = future_df_for_prophet.ds.apply(is_salary_day)
    print(oil.columns)
    print(future_df_for_prophet.columns)
    future_df_for_prophet = future_df_for_prophet.merge(oil,left_on='ds',right_on='date',how="left")[['ds','salary_day','dcoilwtico']]
    future_df_for_prophet['dcoilwtico'].fillna(method='ffill',inplace=True)
    return future_df_for_prophet
def create_future_df_test(stores,items,mindate,maxdate,oil,promotion_perc=0.085):
    df = pd.DataFrame({'date':pd.date_range(mindate,maxdate,freq="D")})
    df.set_index("date")
    df = pd.DataFrame(list(product(df.date.unique(), stores,items)), columns=['date', 'store_nbr','item_nbr'])
    df["onpromotion"] = False
    #randomly set 0.085% of onpromotion to True
    df.loc[df.query('onpromotion == False').sample(frac=promotion_perc).index, 'onpromotion'] = True
    df['salary_day'] = df.date.apply(is_salary_day)
    df= df.merge(oil,on="date",how="left")
    df.dcoilwtico.fillna(method='ffill',inplace=True)
    return df

#%%
train = train.merge(items,on='item_nbr',how='left')
train = train.merge(stores,on='store_nbr',how='left')
#%%
test = test.merge(items,on='item_nbr',how='left')
test = test.merge(stores,on='store_nbr',how='left')
#%%
# calculate item_nbr/item_family ratio for store.city, store.cluster, store.type, store.state and calculate mean and median
ratio_pkl = "input/ratios.pkl"
if not Path(ratio_pkl).is_file():
    item_city = train[['item_nbr','family','city','cluster','state','type','onpromotion','unit_sales']].groupby(['item_nbr','city'],sort=False).agg({'unit_sales':np.sum,'family':'first','cluster':'first','state':'first','type':'first'}).rename(columns={'unit_sales':'item_city_sum'}).reset_index()
    family_city = train[['family','city','unit_sales']].groupby(['family','city'],sort=False).agg({'unit_sales':np.sum}).rename(columns={'unit_sales':'family_city_sum'}).reset_index()
    item_city = item_city.merge(family_city,on=['family','city'],how="left")
    item_city['item_city_ratio']=item_city.item_city_sum/item_city.family_city_sum
    item_cluster = train[['item_nbr','family','cluster','onpromotion','unit_sales']].groupby(['item_nbr','cluster'],sort=False).agg({'unit_sales':np.sum,'family':'first','cluster':'first','state':'first','type':'first'}).rename(columns={'unit_sales':'item_cluster_sum'}).reset_index()
    family_cluster = train[['family','cluster','unit_sales']].groupby(['family','cluster'],sort=False).agg({'unit_sales':np.sum}).rename(columns={'unit_sales':'family_cluster_sum'}).reset_index()
    item_cluster = item_cluster.merge(family_cluster,on=['family','cluster'],how='left')
    item_cluster['item_cluster_ratio'] = item_cluster.item_cluster_sum/item_cluster.family_cluster_sum
    item_state = train[['item_nbr','family','state','onpromotion','unit_sales']].groupby(['item_nbr','state'],sort=False).agg({'unit_sales':np.sum,'family':'first','cluster':'first','state':'first','type':'first'}).rename(columns={'unit_sales':'item_state_sum'}).reset_index()
    family_state = train[['family','state','unit_sales']].groupby(['family','state'],sort=False).agg({'unit_sales':np.sum}).rename(columns={'unit_sales':'family_state_sum'}).reset_index()
    item_state = item_state.merge(family_state,on=['family','state'],how='left')
    item_state['item_state_ratio'] = item_state.item_state_sum/item_state.family_state_sum
    item_type = train[['item_nbr','family','type','onpromotion','unit_sales']].groupby(['item_nbr','type'],sort=False).agg({'unit_sales':np.sum,'family':'first','cluster':'first','state':'first','type':'first'}).rename(columns={'unit_sales':'item_type_sum'}).reset_index()
    family_type = train[['family','type','unit_sales']].groupby(['family','type'],sort=False).agg({'unit_sales':np.sum}).rename(columns={'unit_sales':'family_type_sum'}).reset_index()
    item_type = item_type.merge(family_type,on=['family','type'],how='left')
    item_type['item_type_ratio'] = item_type.item_type_sum/item_type.family_type_sum
    ratio_dumps = cpkl.dumps([item_city,item_cluster,item_state,item_type])
    with open(ratio_pkl,'wb') as f:
        f.write(ratio_dumps)
else:
    print('Loading ratio dump')
    item_city,item_cluster,item_state,item_type = cpkl.loads(open(ratio_pkl,'rb').read())
#%%
#get item_nbrs not covered by the above groupings
#60 items appear in the test set for the first time
if False:
    family_city_uniques = item_city[['family','city']].drop_duplicates()
    family_city_uniques['comb'] = family_city_uniques.family+family_city_uniques.city
    test_family_city_uniques = test[['family','city']].drop_duplicates()
    test_family_city_uniques['comb'] = test_family_city_uniques.family+test_family_city_uniques.city
    family_cluster_uniques = item_cluster[['family','cluster']].drop_duplicates()
    family_cluster_uniques['comb']  = family_cluster_uniques.family+family_cluster_uniques.cluster.apply(str)
    test_family_cluster_uniques = test[['family','cluster']].drop_duplicates()
    test_family_cluster_uniques['comb'] = test_family_cluster_uniques.family+test_family_cluster_uniques.cluster.apply(str)
#%%
# sample data for testing methods
if False:
    samp_stores = [25,99]
    #samp_items = [103665,105575]
    samp = train.loc[(train.store_nbr.isin(samp_stores) )]
    samp['salary_day'] = samp.date.apply(is_salary_day)
    samp = samp.merge(oil,on='date',how="left")
    samp['dcoilwtico'].fillna(method='ffill',inplace=True)
    #samp = samp.merge(items,on="item_nbr")
    #samp = samp.merge(stores,on='store_nbr',how="left")
    samp_family_store =samp.groupby(['date','store_nbr','family']).agg({'unit_sales':np.sum,'salary_day':'first','dcoilwtico':'first','onpromotion':np.max}).reset_index().rename(columns={'unit_sales':'date_family_store_unit_sales'})
    #samp_prophet_mods = samp.groupby(['store_nbr','item_nbr'],sort=False).apply(lambda x : fit_prophet(x,holidays_for_prophet)).to_frame('Model').reset_index()
    
    #test_future = create_future_df_test(samp_stores,samp_items,prediction_window_start_date,prediction_window_end_date,oil)
    #test_future['yhat'] = np.nan
    #test_preds = test_future.groupby(['store_nbr','item_nbr']).apply(lambda x : predictWithData(x,samp_prophet_mods)).reset_index()
    test_allpreds=pd.DataFrame()
    b = False
    for st in samp_stores:
        for it in samp_items:
            df = create_prophet_futuredf(prediction_window_start_date,prediction_window_end_date,oil)
            df['store_nbr'] = st
            df['item_nbr'] = it
            df['onpromotion'] = False
            df.loc[df.query('onpromotion == False').sample(frac=0.085).index, 'onpromotion'] = True
            tmp_pred = predictWithData(df,samp_prophet_mods)
            print ("store number {0}".format(st))
            test_allpreds = pd.concat([test_allpreds,tmp_pred],ignore_index=True)

#    future_df = create_prophet_futuredf(datetime.date(2017,8,16),datetime.date(2017,8,31),oil)
#    future_df['onpromotion'] = False
#    future_df[future_df.ds==datetime.date(2017,8,26)].onpromotion=True
#    preds = prop_mods.groupby(['store_nbr','item_nbr']).apply(lambda x : predict(x,future_df)).reset_index()
    
#%%
# prepare data and model
train['salary_day'] = train.date.apply(is_salary_day)
train = train.merge(oil,on="date",how="left")
train['dcoilwtico'].fillna(method='ffill',inplace=True)
train['onpromotion'] = train.onpromotion.apply(lambda x : 1 if x else 0)
prophet_models = train.groupby(['store_nbr','item_nbr']).apply(lambda x : fit_prophet(x,holidays_for_prophet)).to_frame('Model').reset_index()
# predict
test_preds=pd.DataFrame()
for st in test.store_nbr.unique():
    for it in test.item_nbr.unique():
        df = create_prophet_futuredf(prediction_window_start_date,prediction_window_end_date,oil)
        df['store_nbr'] = st
        df['item_nbr'] = it
        df['onpromotion'] = test[(test.item_nbr==it) &(test.store_nbr==st)]['onpromotion']
        tmp_pred = predictWithData(df,prophet_models)
        print ("store number {0}".format(st))
        test_allpreds = pd.concat([test_allpreds,tmp_pred],ignore_index=True)
