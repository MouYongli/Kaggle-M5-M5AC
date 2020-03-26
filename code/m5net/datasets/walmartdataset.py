import os
import os.path as osp
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

here = osp.dirname(osp.abspath(__file__))
ROOT = osp.join(here, '../../data/')

class WalmartDataset(Dataset):
    def __init__(self, root=None, split='train'):
        if root is not None:
            self.root = root
        else:
            self.root = ROOT
        self.split = split
        self.calendar = pd.read_csv(osp.join(self.root, 'calendar.csv'))
        self.train_validation = pd.read_csv(osp.join(self.root, 'train_validation.csv'))
        self.sell_prices = pd.read_csv(osp.join(self.root, 'sell_prices.csv'))

    def __len__(self):
        return self.train_validation.shape[0]

    def __getitem__(self, index):
        id = self.train_validation.loc[index, :]
        print(id)
        return id



if __name__ == "__main__":
    dataset = WalmartDataset()
    item = dataset[0]


