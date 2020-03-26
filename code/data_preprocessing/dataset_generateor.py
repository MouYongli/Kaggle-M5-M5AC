import os
import os.path as osp
import numpy as np
import pandas as pd
#
# here = osp.dirname(osp.abspath(__file__))
#
# calendar = pd.read_csv('../data/calendar.csv')
sales_train_validation = pd.read_csv('../data/sales_train_validation.csv')
# sell_prices = pd.read_csv('../data/sell_prices.csv')
# # # Initialize daily_prices dataframe
# columns = ['d_%d'%i for i in range(1,1914)]
# daily_prices = pd.DataFrame(columns=['id']+columns)
# daily_prices['id'] = sales_train_validation['id']
# i = 0
# for id in daily_prices['id']:
#     print('%s is started!'%id)
#     store_id = id[-15:-11]
#     item_id = id[:-16]
#     for wm_yr_wk in sell_prices.loc[((sell_prices.store_id == store_id) & (sell_prices.item_id == item_id)), 'wm_yr_wk']:
#         for day in calendar.loc[calendar.wm_yr_wk == wm_yr_wk, 'd']:
#             daily_prices.loc[daily_prices.id == id ,day] =  sell_prices.loc[((sell_prices.store_id == store_id) & (sell_prices.item_id == item_id) & (sell_prices.wm_yr_wk == wm_yr_wk)), 'sell_price']
#     i = i + 1
#     print('%d/30490 is finished!'%i)
# daily_prices.to_csv('./daily_prices.csv', index=False)
idx = np.random.permutation(sales_train_validation.index)

sales_train_validation.reindex(idx).to_csv('./train_validation.csv', index=False)