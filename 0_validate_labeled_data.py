#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:47:02 2020
​
@author: aadityabhatia
"""

import pandas as pd
import argparse

# arg parser
parser = argparse.ArgumentParser(description='Checks the format of the labeled data')
parser.add_argument('-l', '--lab_d', default='reference_data.csv')
args = parser.parse_args()
lab_d = args.lab_d

df = pd.read_csv(lab_d)

check_list = ['query', 'brand', 'product_name', 'product_id',
              'brand_check', 'product_check', 'impressions', 'clicks', 'ctr']
for item in check_list:
    if item not in check_list:
        print(item, 'not found in the csv. Please input correct csv.\nExiting')
        exit()

df['brand_check'] = df.brand_check.replace(to_replace=['N', 'Y', 'n', 'y'], value=[0, 1, 0, 1])
df['product_check'] = df.product_check.replace(to_replace=['N', 'Y', 'n', 'y'], value=[0, 1, 0, 1])

if df.brand_check.nunique() != 2 or df.product_check.nunique() != 2:
    print('Incorrect format in labels')

df = df.brand.dropna()
df.to_csv(lab_d, index=0)
