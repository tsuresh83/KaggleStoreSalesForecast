# -*- coding: utf-8 -*-
import pandas as pd
from datetime import timedelta

dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8'}

train = pd.read_csv('input/train.csv', usecols=[1,2,3,4], dtype=dtypes, parse_dates=['date'],
                    skiprows=range(1, 101688779) #Skip dates before 2017-01-01
                    )
items_master = pd.read_csv("input/items.csv")
train.loc[(train.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
#train['unit_sales'] =  train['unit_sales'].apply(pd.np.log1p) #logarithm conversion
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
train = train.merge(items_master,on='item_nbr',how="left")
lastdate = train.iloc[train.shape[0]-1].date
train_fam_sums = train[['date','store_nbr','family','dow','unit_sales']].groupby(['store_nbr','family','dow']).agg({'unit_sales':sum}).reset_index()
#Load test
test = pd.read_csv('input/test.csv', dtype=dtypes, parse_dates=['date'])
test['dow'] = test['date'].dt.dayofweek
test = test.merge(items_master,on='item_nbr',how="left")
#item family stats
item_family_dw_mean = train_fam_sums[['store_nbr','family','dow','unit_sales']].groupby(['store_nbr','family','dow'])['unit_sales'].mean().to_frame('fam_madw').reset_index()
item_family_wk_mean = item_family_dw_mean[['store_nbr','family','fam_madw']].groupby(['store_nbr', 'family'])['fam_madw'].mean().to_frame('fam_mawk').reset_index()
item_sums = train[['item_nbr','family','unit_sales']].groupby(['item_nbr']).agg({'unit_sales':'sum','family':'first'}).reset_index()
item_sums.rename(columns={'unit_sales':'item_sum'},inplace=True)
family_sums = train[['family','unit_sales']].groupby(['family']).agg({'unit_sales':'sum'}).reset_index()
family_sums.rename(columns={'unit_sales':'family_sum'},inplace=True)
item_family_sums = item_sums.merge(family_sums,on=['family'],how="left")
item_family_sums['item_family_ratio'] = item_family_sums.item_sum/item_family_sums.family_sum
#Days of Week Means
#By tarobxl: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/42948
ma_dw = train[['item_nbr','store_nbr','dow','unit_sales']].groupby(['item_nbr','store_nbr','dow'])['unit_sales'].mean().to_frame('madw')
ma_dw.reset_index(inplace=True)
ma_wk = ma_dw[['item_nbr','store_nbr','madw']].groupby(['store_nbr', 'item_nbr'])['madw'].mean().to_frame('mawk')
ma_wk.reset_index(inplace=True)

#Moving Averages
ma_is = train[['item_nbr','store_nbr','unit_sales']].groupby(['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais226')
ma_is_family = train[['store_nbr','family','unit_sales']].groupby(['store_nbr','family'])['unit_sales'].mean().to_frame('fam_mais226')
for i in [112,56,28,14,7,3,1]:
    tmp = train[train.date>lastdate-timedelta(int(i))]
    tmpg = tmp.groupby(['item_nbr','store_nbr'])['unit_sales'].mean().to_frame('mais'+str(i))
    ma_is = ma_is.join(tmpg, how='left')
    tmpg = tmp.groupby(['store_nbr','family'])['unit_sales'].mean().to_frame('fam_mais'+str(i))
    ma_is_family = ma_is_family.join(tmpg, how='left')

del tmp,tmpg

ma_is['mais']=ma_is.median(axis=1)
ma_is.reset_index(inplace=True)
ma_is_family['fam_mais']=ma_is_family.median(axis=1)
ma_is_family.reset_index(inplace=True)
test = pd.merge(test, ma_is, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_wk, how='left', on=['item_nbr','store_nbr'])
test = pd.merge(test, ma_dw, how='left', on=['item_nbr','store_nbr','dow'])
#item family stats
test_fam = pd.merge(test, ma_is_family, how='left', on=['store_nbr','family'])
test_fam = pd.merge(test_fam, item_family_wk_mean, how='left', on=['store_nbr','family'])
test_fam = pd.merge(test_fam, item_family_dw_mean, how='left', on=['store_nbr','family','dow'])
test_fam = pd.merge(test_fam, item_family_sums, how="left", on=['item_nbr'])
del ma_is, ma_wk, ma_dw, ma_is_family,item_family_wk_mean,item_family_dw_mean

#Forecasting Test
test_fam['unit_sales'] = test_fam.mais
pos_idx = (test_fam['mais'] >= 0) & (test_fam['mawk'] > 0) & (test_fam['madw'] > 0)
#test_pos = test_fam.loc[pos_idx]
#test_zeros = test_fam.loc[~pos_idx]
test_fam.loc[pos_idx, 'unit_sales'] = test_fam.loc[pos_idx,'mais'] * test_fam.loc[pos_idx,'madw'] / test_fam.loc[pos_idx,'mawk']
#test.loc[:, "unit_sales"].fillna(0, inplace=True)
test_fam.loc[~pos_idx,'unit_sales'] = (test_fam.loc[~pos_idx,'fam_mais'] * test_fam.loc[~pos_idx,'fam_madw'] / test_fam.loc[~pos_idx,'fam_mawk'])*test_fam.loc[~pos_idx,'item_family_ratio']
test_fam.loc[:, "unit_sales"].fillna(0, inplace=True)
#test_fam['unit_sales'] = test_fam['unit_sales'].apply(pd.np.expm1) # restoring unit values 

#15% more for promotion items
#By tarobxl: https://www.kaggle.com/tarobxl/overfit-lb-0-532-log-ma
test_fam.loc[test_fam['onpromotion'] == True, 'unit_sales'] = test_fam.loc[test_fam['onpromotion'] == True, 'unit_sales'] * 1.15

test_fam[['id','unit_sales']].to_csv('results/ma8dwof-fixzeros-storefamily-itemratio.csv.gz', index=False, float_format='%.3f', compression='gzip')
# baseline without log
test_fam.loc[~pos_idx,'unit_sales'] = 0
test_fam.loc[:, "unit_sales"].fillna(0, inplace=True)
#test_fam['unit_sales'] = test_fam['unit_sales'].apply(pd.np.expm1) # restoring unit values 

#15% more for promotion items
#By tarobxl: https://www.kaggle.com/tarobxl/overfit-lb-0-532-log-ma
#test_fam.loc[test_fam['onpromotion'] == True, 'unit_sales'] = test_fam.loc[test_fam['onpromotion'] == True, 'unit_sales'] * 1.15

test_fam[['id','unit_sales']].to_csv('results/ma8dwof-baselinewithoutlog-.csv.gz', index=False, float_format='%.3f', compression='gzip')

