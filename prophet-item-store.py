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
from datetime import timedelta
from functools import reduce
from fbprophet import Prophet
import multiprocessing as mp
from itertools import repeat
from joblib import Parallel, delayed
import calendar
#%% 
#df_train = pd.read_csv(
#    'input/train.csv', usecols=[1, 2, 3, 4, 5], dtype={'onpromotion': str},
#    converters={'unit_sales': lambda u: float(u) if float(u) > 0 else 0},
#    skiprows=range(1, 124035460)
#)
script_name='prophet_item_store'
train_pkl_file = 'input/train_pkl'
test_pkl_file = 'input/test_pkl'
models_pkl = 'results/item_store_mdl_prophet_pkl'
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
big_bang = datetime.date(2014,12,31)
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
    m = Prophet(holidays=holidays,uncertainty_samples=0)
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
#this is the same as fit_prophet but returns a dataframe - for parallelization
def fit_prophet_1(data,holidays):
    if(data.shape[0]<3):
        return np.nan
    m = Prophet(holidays=holidays,uncertainty_samples=0)
    #m = Prophet()
    # add additional regressors
    m.add_regressor('salary_day')
    m.add_regressor('onpromotion')
    m.add_regressor('dcoilwtico')
    store_nbr_local = data.store_nbr.unique()
    item_nbr_local = data.item_nbr.unique()
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
    tmp_df = pd.DataFrame({'store_nbr':store_nbr_local,'item_nbr':item_nbr_local,'Model':m})
    return tmp_df
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
    
#    print("{0},{1}".format(len(modelRow),len(data)))
#    print(modelRow)
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
# sample data for testing methods
if False:
    samp_stores = [25,99]
    samp_items = [103665,105575]
    samp = train.loc[(train.store_nbr.isin(samp_stores) ) & (train.item_nbr.isin(samp_items))]
    samp['salary_day'] = samp.date.apply(is_salary_day)
    samp = samp.merge(oil,on='date',how="left")
    samp['dcoilwtico'].fillna(method='ffill',inplace=True)
    #samp_prophet_mods = samp.groupby(['store_nbr','item_nbr'],sort=False).apply(lambda x : fit_prophet(x,holidays_for_prophet)).to_frame('Model').reset_index()
    samp_prophet_gps = samp.groupby(['store_nbr','item_nbr'])
    samp_prophet_mods = len(samp_prophet_gps.groups)*[None]
    ctr = 0
    parallel = False
    if parallel:
        #pool = mp.Pool(processes=4)
        #samp_prophet_mods = pool.starmap(fit_prophet, zip(samp_prophet_gps, repeat(holidays_for_prophet)))
        samp_prophet_mods = Parallel(n_jobs=4)(delayed(fit_prophet_1)(group,holidays_for_prophet) for name, group in samp_prophet_gps)
    else:
        for name, gp in samp_prophet_gps:
            print(name)
            tmp_mod = fit_prophet(gp,holidays_for_prophet)
            tmp_df = pd.DataFrame({'store_nbr':[name[0]],'item_nbr':[name[1]],'Model':[tmp_mod]})
            #samp_prophet_mods = pd.concat([samp_prophet_mods,tmp_df],ignore_index=True)
            #samp_prophet_mods.append(tmp_df)
            samp_prophet_mods[ctr] = tmp_df
            ctr+=1
    
    samp_prophet_mods = pd.concat(samp_prophet_mods,axis=0)
    #test_future = create_future_df_test(samp_stores,samp_items,prediction_window_start_date,prediction_window_end_date,oil)
    #test_future['yhat'] = np.nan
    #test_preds = test_future.groupby(['store_nbr','item_nbr']).apply(lambda x : predictWithData(x,samp_prophet_mods)).reset_index()
    test_allpreds=8*[None]
    b = False
    ctr=0
    for st in samp_stores:
        for it in samp_items:
            df = create_prophet_futuredf(prediction_window_start_date,prediction_window_end_date,oil)
            df['store_nbr'] = st
            df['item_nbr'] = it
            df['onpromotion'] = False
            df.loc[df.query('onpromotion == False').sample(frac=0.085).index, 'onpromotion'] = True
            tmp_pred = predictWithData(df,samp_prophet_mods)
            print ("store number {0}".format(st))
            #test_allpreds = pd.concat([test_allpreds,tmp_pred],ignore_index=True)
            test_allpreds[ctr] = tmp_pred
            ctr+=1
    test_allpreds = pd.concat(test_allpreds,axis=0)
    for_plot = test_allpreds.dropna(axis=0)
    samp_in_august = samp[samp.date>datetime.date(2017,7,31)][['date','store_nbr','item_nbr','unit_sales']]
    for_plot1 = for_plot[['ds','store_nbr','item_nbr','yhat']]
    for_plot1.rename(columns={'ds':'date','yhat':'unit_sales'},inplace=True)
    for_plot1 = pd.concat([for_plot1,samp_in_august],ignore_index=True,axis=0)
    plt.plot(for_plot[for_plot.item_nbr==103665].ds,for_plot[for_plot.item_nbr==103665].yhat)
    plt.plot(for_plot[for_plot.item_nbr==105575].ds,for_plot[for_plot.item_nbr==105575].yhat)
    plt.plot(for_plot1[for_plot1.item_nbr==103665].date,for_plot1[for_plot1.item_nbr==103665].unit_sales)
    plt.plot(for_plot1[for_plot1.item_nbr==105575].date,for_plot1[for_plot1.item_nbr==105575].unit_sales)
    samp_in_august_prev = samp[(samp.date>datetime.date(2016,7,31)) &
                          (samp.date<datetime.date(2016,9,1))][['date','store_nbr','item_nbr','unit_sales']]
    plt.plot(samp_in_august_prev[samp_in_august_prev.item_nbr==103665].date,samp_in_august_prev[samp_in_august_prev.item_nbr==103665].unit_sales)
    plt.plot(samp_in_august_prev[samp_in_august_prev.item_nbr==105575].date,samp_in_august_prev[samp_in_august_prev.item_nbr==105575].unit_sales)
#    future_df = create_prophet_futuredf(datetime.date(2017,8,16),datetime.date(2017,8,31),oil)
#    future_df['onpromotion'] = False
#    future_df[future_df.ds==datetime.date(2017,8,26)].onpromotion=True
#    preds = prop_mods.groupby(['store_nbr','item_nbr']).apply(lambda x : predict(x,future_df)).reset_index()
    # compare the test prediction against item store means
    ma_test = pd.read_csv("results/ma8dwof.csv")
    ma_test = ma_test.merge(test,on="id",how="left")
    ma_test = ma_test[(ma_test.store_nbr.isin(samp_stores) )&(ma_test.item_nbr.isin(samp_items))]
    ma_test1 = ma_test[['date','store_nbr','item_nbr','unit_sales']]
    ma_test1 = pd.concat([ma_test1,samp_in_august],ignore_index=True,axis=0)
    plt.plot(ma_test[ma_test.item_nbr==103665].date,ma_test[ma_test.item_nbr==103665].unit_sales)
    plt.plot(ma_test[ma_test.item_nbr==105575].date,ma_test[ma_test.item_nbr==105575].unit_sales)
    plt.plot(ma_test1[ma_test1.item_nbr==103665].date,ma_test1[ma_test1.item_nbr==103665].unit_sales)
    plt.plot(ma_test1[ma_test1.item_nbr==105575].date,ma_test1[ma_test1.item_nbr==105575].unit_sales)
#%%
# prepare data and model
train['salary_day'] = train.date.apply(is_salary_day)
train = train.merge(oil,on="date",how="left")
train['dcoilwtico'].fillna(method='ffill',inplace=True)
train['onpromotion'] = train.onpromotion.apply(lambda x : 1 if x else 0)
#prophet_models = train.groupby(['store_nbr','item_nbr']).apply(lambda x : fit_prophet(x,holidays_for_prophet)).to_frame('Model').reset_index()
prophet_models = pd.DataFrame()
store_item_gps = train.groupby(['store_nbr','item_nbr'])
ctr = 0
#prophet_mods = len(store_item_gps)*[None]
#for name, gp in store_item_gps:
#    print("fitting....",ctr)
#    tmp_mod = fit_prophet(gp,holidays_for_prophet)
#    tmp_df = pd.DataFrame({'store_nbr':[name[0]],'item_nbr':[name[1]],'Model':[tmp_mod]})
#    #prophet_mods.append(tmp_df)
#    prophet_mods[ctr]=tmp_df
#    ctr+=1
prophet_mods = Parallel(n_jobs=6,verbose=1)(delayed(fit_prophet_1)(group,holidays_for_prophet) for name, group in store_item_gps)
prophet_mods = [x for x in prophet_mods if str(x) != 'nan']
prophet_mods = pd.concat(prophet_mods, axis=0)
#with open(models_pkl,'wb') as f:
#    f.write(cpkl.dumps(prophet_mods))
# predict
test['salary_day'] = test.date.apply(is_salary_day)
test = test.merge(oil,on="date",how="left")
test['dcoilwtico'].fillna(method='ffill',inplace=True)
test['onpromotion'] = test.onpromotion.apply(lambda x : 1 if x else 0)
test_gps = test.groupby(['store_nbr','item_nbr'])
test_allpreds=len(test_gps)*[None]
ctr=0
strn=0
itemn = 0
for name, gp in test_gps:
    tmp_pred = predictWithData(gp,prophet_mods)
    if name[0] !=strn:
        strn+=1
        itemn=0
    itemn+=1     
    print ("predicting {0},{1}:{2}>>{3}...".format(name[0],name[1],strn,itemn))
    #test_allpreds.append(tmp_pred)
    test_allpreds[ctr] = tmp_pred
    ctr+=1
#test_allpreds = Parallel(n_jobs=2,verbose=1)(delayed(predictWithData)(group,prophet_mods) for name, group in test_gps)
test_allpreds = pd.concat(test_allpreds,axis=0)
final = test.merge(test_allpreds,left_on=['date','store_nbr','item_nbr'],right_on=['ds','store_nbr','item_nbr'])
final.rename(columns={'yhat':'unit_sales'},inplace=True)
final.loc[:, "unit_sales"].fillna(0, inplace=True)
final[['id','unit_sales']].to_csv('results/'+script_name+"_"+str(big_bang+timedelta(1))+".csv", index=False, float_format='%.3f', compression='gzip')


#for st in test.store_nbr.unique():
#    for it in test.item_nbr.unique():
#        df = create_prophet_futuredf(prediction_window_start_date,prediction_window_end_date,oil)
#        df['store_nbr'] = st
#        df['item_nbr'] = it
#        df['onpromotion'] = test[(test.item_nbr==it) &(test.store_nbr==st)]['onpromotion']
#        tmp_pred = predictWithData(df,samp_prophet_mods)
#        print ("store number {0}".format(st))
#        test_allpreds = pd.concat([test_allpreds,tmp_pred],ignore_index=True)
