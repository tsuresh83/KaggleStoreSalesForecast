# -*- coding: utf-8 -*-
import pandas as pd
from datetime import timedelta

dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8'}

train = pd.read_csv('input/train.csv', usecols=[1,2,3,4], dtype=dtypes, parse_dates=['date'],
                    skiprows=range(1, 101688779) #Skip dates before 2017-01-01
                    )
stores = pd.read_csv("input/stores.csv")
train.loc[(train.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
train['unit_sales'] =  train['unit_sales'].apply(pd.np.log1p) #logarithm conversion
train['dow'] = train['date'].dt.dayofweek

# creating records for all items, in all markets on all dates
# for correct calculation of daily unit sales averages.
u_dates = train.date.unique()
u_stores = train.store_nbr.unique()
u_items = train.item_nbr.unique()
train.set_index(['date', 'store_nbr', 'item_nbr'], inplace=True)
train = train.reindex(
    pd.MultiIndex.from_product(
        (u_dates, u_stores, u_items),
        names=['date','store_nbr','item_nbr']
    )
)

del u_dates, u_stores, u_items

train.loc[:, 'unit_sales'].fillna(0, inplace=True) # fill NaNs
train.reset_index(inplace=True) # reset index and restoring unique columns  
train = train.merge(stores,on='store_nbr',how="left")
lastdate = train.iloc[train.shape[0]-1].date

#Load test
test = pd.read_csv('input/test.csv', dtype=dtypes, parse_dates=['date'])
test['dow'] = test['date'].dt.dayofweek
test = test.merge(stores,on='store_nbr',how="left")
#item family stats
item_cluster_dw_mean = train[['item_nbr','cluster','dow','unit_sales']].groupby(['item_nbr','cluster','dow'])['unit_sales'].mean().to_frame('cluster_madw').reset_index()
item_cluster_wk_mean = item_cluster_dw_mean[['item_nbr','cluster','cluster_madw']].groupby(['item_nbr', 'cluster'])['cluster_madw'].mean().to_frame('cluster_mawk').reset_index()
#Days of Week Means
#By tarobxl: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/42948
ma_dw = train[['item_nbr','store_nbr','dow','unit_sales']].groupby(['item_nbr','store_nbr','dow'])['unit_sales'].mean().to_frame('madw')
ma_dw.reset_index(inplace=True)
ma_wk = ma_dw[['item_nbr','store_nbr','madw']].groupby(['store_nbr', 'item_nbr'])['madw'].mean().to_frame('mawk')
ma_wk.reset_index(inplace=True)

#Moving Averages
ma_is = train[['item_nbr','store_nbr','unit_sales']].groupby(['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais226')
ma_is_cluster = train[['item_nbr','cluster','unit_sales']].groupby(['item_nbr','cluster'])['unit_sales'].mean().to_frame('cluster_mais226')
for i in [112,56,28,14,7,3,1]:
    tmp = train[train.date>lastdate-timedelta(int(i))]
    tmpg = tmp.groupby(['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais'+str(i))
    ma_is = ma_is.join(tmpg, how='left')
    tmpg = tmp.groupby(['item_nbr','cluster'])['unit_sales'].mean().to_frame('cluster_mais'+str(i))
    ma_is_cluster = ma_is_cluster.join(tmpg, how='left')

del tmp,tmpg

ma_is['mais']=ma_is.median(axis=1)
ma_is.reset_index(inplace=True)
ma_is_cluster['cluster_mais']=ma_is_cluster.median(axis=1)
ma_is_cluster.reset_index(inplace=True)
test = pd.merge(test, ma_is, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_wk, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_dw, how='left', on=['item_nbr','store_nbr','dow'])
#item family stats
test_fam = pd.merge(test, ma_is_cluster, how='left', on=['item_nbr','cluster'])
test_fam = pd.merge(test_fam, item_cluster_wk_mean, how='left', on=['item_nbr','cluster'])
test_fam = pd.merge(test_fam, item_cluster_dw_mean, how='left', on=['item_nbr','cluster','dow'])
del ma_is, ma_wk, ma_dw, ma_is_cluster,item_cluster_wk_mean,item_cluster_dw_mean

#Forecasting Test
test_fam['unit_sales'] = test_fam.mais
pos_idx = (test_fam['mais'] > 0) & (test_fam['mawk'] > 0) & (test_fam['madw'] > 0)
test_pos = test_fam.loc[pos_idx]
test_zeros = test_fam.loc[~pos_idx]
test_fam.loc[pos_idx, 'unit_sales'] = test_pos['mais'] * test_pos['madw'] / test_pos['mawk']
#test.loc[:, "unit_sales"].fillna(0, inplace=True)
test_fam.loc[~pos_idx,'unit_sales'] = test_zeros['cluster_mais'] * test_zeros['cluster_madw'] / test_zeros['cluster_mawk']
test_fam.loc[:, "unit_sales"].fillna(0, inplace=True)
test_fam['unit_sales'] = test_fam['unit_sales'].apply(pd.np.expm1) # restoring unit values 

#15% more for promotion items
#By tarobxl: https://www.kaggle.com/tarobxl/overfit-lb-0-532-log-ma
test_fam.loc[test_fam['onpromotion'] == True, 'unit_sales'] = test_fam.loc[test_fam['onpromotion'] == True, 'unit_sales'] * 1.15

test_fam[['id','unit_sales']].to_csv('ma8dwof-fixzeros-meanitem-cluster.csv.gz', index=False, float_format='%.3f', compression='gzip')
len(test_fam[test_fam.unit_sales==0])