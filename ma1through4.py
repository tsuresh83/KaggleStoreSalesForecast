#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:54:26 2017

@author: suresh
"""

import pandas as pd
import numpy as np
ma_item_store = pd.read_csv("/media/3TB/kaggle/salesforecast/ma8dwof.csv")
ma_itemfamily_store = pd.read_csv("/media/3TB/kaggle/salesforecast/ma8dwof-fixzeros.csv")
ma_item_city = pd.read_csv("/media/3TB/kaggle/salesforecast/ma8dwof-fixzeros-meanitem-city.csv")
ma_item_cluster = pd.read_csv("/media/3TB/kaggle/salesforecast/ma8dwof-fixzeros-meanitem-cluster.csv")
sub = ma_item_store.merge(ma_itemfamily_store,on="id")
sub = sub.merge(ma_item_city,on="id")
sub = sub.merge(ma_item_cluster,on="id")
sub['means'] = sub[['unit_sales_x', 'unit_sales_y', 'unit_sales_x', 'unit_sales_y']].mean(axis=1)
sub.rename(columns={'means':'unit_sales'},inplace=True)
sub[['id','unit_sales']].to_csv('mean-item-store-family-city-cluster.csv.gz', index=False, float_format='%.3f', compression='gzip')
