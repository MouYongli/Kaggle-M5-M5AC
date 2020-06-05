## This Python 3 environment comes with many helpful analytics libraries installed
## It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
## For example, here's several helpful packages to load
import os
import os.path as osp
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tqdm
import gc
import seaborn as sns
from sklearn.preprocessing import LabelEncoder



import datetime
import pytz
start_time = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

## Input data files are available in the read-only "../input/" directory
## For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
## import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
## You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
## You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

here = osp.dirname(osp.abspath(__file__))
base_dir = osp.join(here, 'data')
output_dir = here
train_dir = osp.join(base_dir, 'sales_train_validation.csv')
test_dir = osp.join(base_dir, 'sample_submission.csv')
calendar_dir = osp.join(base_dir, 'calendar.csv')
price_dir = osp.join(base_dir, 'sell_prices.csv')

df_train = pd.read_csv(train_dir)
df_test = pd.read_csv(test_dir)
df_calendar = pd.read_csv(calendar_dir)
df_price = pd.read_csv(price_dir)

df_product = df_train[['item_id', 'store_id']]

## Reduce memory usage
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

## Making train data
def making_train_data(df_train):
    print("processing train data")
    df_train_after = pd.melt(df_train, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name='days', value_name='demand')
    df_train_after['days'] = df_train_after['days'].map(lambda x: int(x[2:]))
    df_train_after = df_train_after.drop(['id'], axis=1)
    df_train_after = reduce_mem_usage(df_train_after)
    gc.collect()
    return df_train_after

## Making test data
def making_test_data(df_test):
    print("processing test data")
    df_test['item_id'] = df_test['id'].map(lambda x: x[:-16])
    df_test['dept_id'] = df_test['item_id'].map(lambda x: x[:-4])
    df_test['cat_id'] = df_test['dept_id'].map(lambda x: x[:-2])
    df_test['store_id'] = df_test['id'].map(lambda x: x[-15:-11])
    df_test['state_id'] = df_test['store_id'].map(lambda x: x[:-2])
    df_test['va_or_ev'] = df_test['id'].map(lambda x: x[-10:])
    df_test_val = df_test.loc[df_test['va_or_ev'] == 'validation', :]
    df_test_ev = df_test.loc[df_test['va_or_ev'] == 'evaluation', :]
    df_test_val_after = pd.melt(df_test_val, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'va_or_ev'], var_name='days', value_name='demand')
    df_test_ev_after = pd.melt(df_test_ev, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'va_or_ev'], var_name='days', value_name='demand')
    df_test_after = pd.concat([df_test_val_after, df_test_ev_after])
    df_test_after['days'] = df_test_after['days'].map(lambda x: int(x[1:]))
    df_test_after.loc[df_test_after['va_or_ev']=='evaluation', ['days']] += 28
    df_test_after['days'] += 1913
    df_test_after = df_test_after.drop(['va_or_ev'], axis=1)
    df_test_after = df_test_after.drop(['id'], axis=1)
    df_test_after = reduce_mem_usage(df_test_after)
    return df_test_after

## Making train and test data
def making_train_test_data(df_train ,df_test):
    df_train = making_train_data(df_train)
    df_test = making_test_data(df_test)
    return df_train, df_test

## Making calendar data
def making_calendar_data(df_calendar):
    df_calendar = reduce_mem_usage(df_calendar)
    gc.collect()
    print("processing calendar data")
    df_calendar['days'] = df_calendar['d'].map(lambda x: int(x[2:]))
    event_name =  {np.nan: 0,  'Halloween': 1, 'LentStart': 2, "Mother's day": 3, 'Cinco De Mayo': 4, 'EidAlAdha': 5,
                        'SuperBowl': 6, 'IndependenceDay': 7, 'StPatricksDay': 8, 'NBAFinalsEnd': 9, 'Easter': 10,
                        'MemorialDay': 11, 'ValentinesDay': 12, 'MartinLutherKingDay': 13, 'Christmas': 14,
                        'Purim End': 15, 'OrthodoxEaster': 16, 'Thanksgiving': 17, 'ColumbusDay': 18,
                        'VeteransDay': 19,
                        'NBAFinalsStart': 20, 'Pesach End': 21, 'LaborDay': 22, 'Chanukah End': 23,
                        'Eid al-Fitr': 24,
                        'LentWeek2': 25, 'NewYear': 26, 'PresidentsDay': 27, "Father's day": 28,
                        'OrthodoxChristmas': 29,
                        'Ramadan starts': 30}
    event_type = {np.nan: 0, 'Sporting': 1, 'Cultural': 2, 'National': 3, 'Religious': 4}
    df_calendar['event_name_1'] = df_calendar['event_name_1'].map(event_name)
    df_calendar['event_name_2'] = df_calendar['event_name_2'].map(event_name)
    df_calendar['event_type_1'] = df_calendar['event_type_1'].map(event_type)
    df_calendar['event_type_2'] = df_calendar['event_type_2'].map(event_type)
    df_calendar['year'] =  df_calendar['year'].map(lambda x: int(x - 2010))
    df_calendar = df_calendar.drop(['d', 'weekday', 'date'], axis=1)
    df_calendar = reduce_mem_usage(df_calendar)
    gc.collect()
    return df_calendar

## Making sell price data
def making_price_data(df_price):
    print("processing price data")
    df_price = reduce_mem_usage(df_price)
    gc.collect()
    return df_price

def concat_data(df_train, df_test, df_calendar, df_price):
    df_train, df_test = making_train_test_data(df_train ,df_test)
    df_calendar = making_calendar_data(df_calendar)
    df_price = making_price_data(df_price)
    print("concat data")
    df_train = pd.merge(df_train, df_calendar, on='days', how='left')
    df_test = pd.merge(df_test, df_calendar, on='days', how='left')
    df_train = pd.merge(df_train, df_price, on=['wm_yr_wk', 'store_id', 'item_id'], how='left')
    df_test = pd.merge(df_test, df_price, on=['wm_yr_wk', 'store_id', 'item_id'], how='left')
    df_train = df_train.drop(['wm_yr_wk'], axis=1)
    df_test = df_test.drop(['wm_yr_wk'], axis=1)
    del df_calendar, df_price
    gc.collect()
    df_train = reduce_mem_usage(df_train)
    df_test = reduce_mem_usage(df_test)
    gc.collect()
    return df_train, df_test

def labeling_data(df_train, df_test, df_calendar, df_price):
    df_train, df_test = concat_data(df_train, df_test, df_calendar, df_price)
    print("labeling data")
    df_train['item_store'] = df_train['item_id'] + '_' + df_train['store_id']
    df_test['item_store'] = df_test['item_id'] + '_' + df_test['store_id']
    label_columns = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    for c in label_columns:
        le  = LabelEncoder()
        le.fit(df_train[c])
        df_train[c] = le.transform(df_train[c])
        df_test[c] = le.transform(df_test[c])
        if c != 'item_id':
            print(le.classes_)
    df_train = reduce_mem_usage(df_train)
    df_test = reduce_mem_usage(df_test)
    gc.collect()
    return df_train, df_test

df_train, df_test = labeling_data(df_train, df_test, df_calendar, df_price)

df_train['available'] = df_train['sell_price'].map(lambda x: 0 if pd.isnull(x) else 1)
df_test['available'] = df_test['sell_price'].map(lambda x: 0 if pd.isnull(x) else 1)
df_train = df_train.fillna(0)
df_test = df_test.fillna(0)
df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)
gc.collect()

df_train.to_csv(osp.join(output_dir, 'train.csv'), index=False)
df_test.to_csv(osp.join(output_dir, 'test.csv'), index=False)

elapsed_time = (datetime.datetime.now(pytz.timezone('Asia/Tokyo')) - start_time).total_seconds()
print('Elapsed time for making train and test data', elapsed_time)
print('Length of train dataframe: ', len(df_train))
print('Columns of train dataframe:')
for col in df_train.columns:
    print('    ', col)