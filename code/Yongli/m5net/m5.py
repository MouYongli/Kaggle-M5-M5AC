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

import torch
import torch.nn as nn
from torch.utils.data import Dataset 
from torch.autograd import Variable

import datetime
import pytz

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
output_dir = osp.join(here, 'log')

product_dir = osp.join(base_dir, 'sales_train_validation.csv')
train_dir = osp.join(here, 'train.csv')
test_dir = osp.join(here, 'test.csv')

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

class M5Dataset(Dataset):
    def __init__(self, train_dir, test_dir, product_dir, split='train', transform=True):
        super(M5Dataset, self).__init__()
        self.split = split
        self._transform = transform

        start_time = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        self.df_train = pd.read_csv(train_dir)
        self.df_test = pd.read_csv(test_dir)
        self.df_product = pd.read_csv(product_dir)
        self.df_product = self.df_product[['item_id', 'store_id']]
        self.df_train = reduce_mem_usage(self.df_train)
        self.df_test = reduce_mem_usage(self.df_test)
        gc.collect()
        elapsed_time = (datetime.datetime.now(pytz.timezone('Asia/Tokyo')) - start_time).total_seconds()
        print('Elapsed time for making train and test data', elapsed_time)
        print('Length of train dataframe: ', len(self.df_train))
        print('Columns of train dataframe:')
        for col in self.df_train.columns:
        	print('    ', col)

    
    def __len__(self):
        return len(self.df_product)
    
    def __getitem__(self, idx):
        item_id = self.df_product['item_id'][idx]
        store_id = self.df_product['store_id'][idx]
        state = store_id[-4:-2]
        if self.split == 'train':            
            data = self.df_train[self.df_train['item_store'] == item_id + '_' + store_id][-1000:]
        else:
            train_data = self.df_train[self.df_train['item_store'] == item_id + '_' + store_id][-72:]
            test_data = self.df_test[self.df_test['item_store'] == item_id + '_' + store_id]
            data = pd.concat([train_data, test_data])
        embedding_data = data[['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_name_2', 'event_type_1', 'event_type_2']]
        feat_data = data[['wday', 'month', 'year', 'snap_'+state, 'sell_price', 'available']]
        target = data[['demand']] 
        ## 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_name_2', 'event_type_1', 'event_type_2'
        ## 'wday', 'month', 'year', 'snap_'+state, 'sell_price', 'available'
        ## 'demand'
        embedding_data, feat_data, target = embedding_data.to_numpy(), feat_data.to_numpy(), target.to_numpy() 
        if self._transform:
            embedding_data, feat_data, target = self.transform(embedding_data, feat_data, target)
        return item_id, store_id, embedding_data, feat_data, target
        
    def transform(self, embedding_data, feat_data, target):
        embedding_data = embedding_data.astype(np.long)
        embedding_data = torch.from_numpy(embedding_data).long()
        feat_data = feat_data.astype(np.float64)
        feat_data = torch.from_numpy(feat_data).float()
        target = target.astype(np.float64)
        target = torch.from_numpy(target).float()
        return embedding_data, feat_data, target

class M5Net(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layer, seq_length, output_dim=1, batch_first=True):
        super(M5Net, self).__init__()
        self.embedding_dim =embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.batch_first = batch_first
        self.item_embedding = nn.Embedding(3049, embedding_dim)
        self.dept_embedding = nn.Embedding(7, embedding_dim)
        self.cat_embedding = nn.Embedding(3, embedding_dim)
        self.store_embedding = nn.Embedding(10, embedding_dim)
        self.state_embedding = nn.Embedding(3, embedding_dim)
        self.event_name_embedding = nn.Embedding(31, embedding_dim)
        self.event_type_embedding = nn.Embedding(5, embedding_dim)
        self.calendar_prices_feat = nn.Linear(6, embedding_dim)
        self.encode = nn.Linear(4*embedding_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layer, batch_first=batch_first)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x1, x2):
        item = self.item_embedding(x1[:,:,0])
        dept = self.dept_embedding(x1[:,:,1])
        cat = self.cat_embedding(x1[:,:,2])
        store = self.store_embedding(x1[:,:,3])
        state = self.state_embedding(x1[:,:,4])
        en1 = self.event_name_embedding(x1[:,:,5])
        en2 = self.event_name_embedding(x1[:,:,6])
        et1 = self.event_type_embedding(x1[:,:,7])
        et2 = self.event_type_embedding(x1[:,:,8])
        i = item + dept + cat
        s = store + state
        e = en1 + en2 + et1 + et2
        c = self.calendar_prices_feat(x2)
        x = self.encode(torch.cat((i, s, e, c), 2))
        y, (_, _) = self.lstm(x)
        out = self.classifier(y)
        return out

## Define hyper parameters
embedding_dim = 128
hidden_dim = 256
n_layer = 4
output_dim = 1
seq_length = 100
batch_size = 8

epochs = 10
lr = 0.0001
weight_decay = 0.0005

cuda = torch.cuda.is_available()
print('Cuda available: ', cuda)
torch.manual_seed(1337)
if cuda:
    torch.cuda.manual_seed(1337)

## Define data loader
# kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(M5Dataset(train_dir, test_dir, product_dir, split='train', transform=True), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(M5Dataset(train_dir, test_dir, product_dir, split='valid', transform=True), batch_size=batch_size, shuffle=False)

## Define neural network model
model = M5Net(embedding_dim, hidden_dim, n_layer, seq_length, output_dim)
if cuda:
	model = model.cuda()
        
# Define optimizer
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Define loss function
loss_fn = nn.MSELoss()


for epoch in range(epochs):
    if not osp.exists(osp.join(output_dir, 'valid_%d.csv'%epoch)):
        with open('valid_%d.csv'%epoch, 'w') as f:
            header = ['id'] + ['F%d'%i for i in range(1,29)]
            header = map(str, header)
            f.write(','.join(header) + '\n')
    if not osp.exists(osp.join(output_dir, 'eval_%d.csv'%epoch)):
        with open('eval_%d.csv'%epoch, 'w') as f:
            header = ['id'] + ['F%d'%i for i in range(1,29)]
            header = map(str, header)
            f.write(','.join(header) + '\n')
            
    model.train()
    for batch_idx, (item_id, store_id, embedding_data, feat_data, target) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), ncols=80, leave=False):
        assert model.training
        if cuda:
            embedding_data, feat_data, target = embedding_data.cuda(), feat_data.cuda(), target.cuda()
        embedding, feat_data, target = Variable(embedding_data), Variable(feat_data), Variable(target)
        for i in range(19):
            optim.zero_grad()
            predict = model(embedding_data[:,50*i:50*i+100,:], feat_data[:,50*i:50*i+100,:])
            loss = loss_fn(predict, target[:,50*i:50*i+100,:])
            loss.backward()
            optim.step()
            
    checkpoint = {
        'model': model,
        'optim_state_dict': optim.state_dict(),
        'model_state_dict': model.state_dict()
    }
    
    torch.save(checkpoint, 'checkpoint%d.pth.tar'%epch)
    model.eval()
    for batch_idx, (item_id, store_id, embedding_data, feat_data, target) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), ncols=80, leave=False):
        if cuda:
            embedding_data, feat_data, target = embedding_data.cuda(), feat_data.cuda(), target.cuda()
        embedding, feat_data, target = Variable(embedding_data), Variable(feat_data), Variable(target)
        with torch.no_grad():
            predicts = model(embedding_data[:,:100,:], feat_data[:,:100,:])
            predicts = predicts.data.cpu().numpy()[:,-28:,:].squeeze(axis=2)
            with open(osp.join(output_dir, 'valid_%d.csv'%epoch), 'a') as f:
                for (item, store, pred) in zip(item_id, store_id, predicts):
                    log = [item+'_'+store+'_validation'] + [p for p in pred]
                    log = map(str, log)
                    f.write(','.join(log) + '\n')
        with torch.no_grad():
            predicts = model(embedding_data[:,-100:,:], feat_data[:,-100:,:])
            predicts = predicts.data.cpu().numpy()[:,-28:,:].squeeze(axis=2)
            with open(osp.join(output_dir, 'eval_%d.csv'%epoch), 'a') as f:
                for (item, store, pred) in zip(item_id, store_id, predicts):
                    log = [item+'_'+store+'_evaluation'] + [p for p in pred]
                    log = map(str, log)
                    f.write(','.join(log) + '\n')    