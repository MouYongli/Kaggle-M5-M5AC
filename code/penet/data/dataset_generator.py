import numpy as np
import pandas as pd

# here = osp.dirname(osp.abspath(__file__))
# import sys
# sys.path.append('../../')

states = ['CA', 'TX', 'WI']
stores = ['CA_1', 'CA_2', 'CA_3',  'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
categories = ['HOBBIES', 'HOUSEHOLD', 'FOODS']
departments = ['HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2', 'FOODS_1', 'FOODS_2', 'FOODS_3']
sales_train_validation = pd.read_csv('sales_train_validation.csv')
train_validation = pd.DataFrame(columns=['id', 'p_id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])
p_id_df = pd.DataFrame(np.array([i for i in range(len(sales_train_validation))]), columns=['p_id'])
items = sales_train_validation['item_id'].unique().tolist()

train_validation['id'] = sales_train_validation['id']
train_validation['p_id'] = p_id_df['p_id']
train_validation.loc[:,'item_id'] = sales_train_validation.apply(lambda row: items.index(row['item_id']),axis=1)
train_validation.loc[:,'dept_id'] = sales_train_validation.apply(lambda row: departments.index(row['dept_id']),axis=1)
train_validation.loc[:,'cat_id'] = sales_train_validation.apply(lambda row: categories.index(row['cat_id']),axis=1)
train_validation.loc[:,'store_id'] = sales_train_validation.apply(lambda row: stores.index(row['store_id']),axis=1)
train_validation.loc[:,'state_id'] = sales_train_validation.apply(lambda row: states.index(row['state_id']),axis=1)

train_validation.to_csv('./train_validation_sorted.csv', index=False)
idx = np.random.permutation(train_validation.index)
train_validation.reindex(idx).to_csv('./train_validation.csv', index=False)

