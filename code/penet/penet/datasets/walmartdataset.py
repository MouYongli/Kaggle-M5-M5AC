import os.path as osp
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

here = osp.dirname(osp.abspath(__file__))

ROOT_DIR = osp.join(here, '../../data/')

class WalmartDataset(Dataset):
    def __init__(self, root_dir=None, split='train', transform=False):
        super(WalmartDataset, self).__init__()
        if root_dir is not None:
            self.root_dir = root_dir
        else:
            self.root_dir = ROOT_DIR
        self.split = split
        self.transform = transform
        self.train_validation = pd.read_csv(osp.join(self.root_dir, 'train_validation.csv'))

    def __len__(self):
        return self.train_validation.shape[0]

    def __getitem__(self, idx):
        # timestamp_start = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        row = self.train_validation.loc[idx, :]
        id = row['id']
        ## one-hot encoding
        p_id = np.zeros(30490)
        p_id[row['p_id']] = 1

        # item_id = np.zeros(3049)
        # dept_id = np.zeros(7)
        # cat_id = np.zeros(3)
        # store_id = np.zeros(10)
        # state_id = np.zeros(3)
        # item_id[row['item_id']] = 1
        # dept_id[row['dept_id']] = 1
        # cat_id[row['cat_id']] = 1
        # store_id[row['store_id']] = 1
        # state_id[row['state_id']] = 1
        item_id = np.array([row['item_id']])
        dept_id = np.array(row['dept_id'])
        cat_id = np.array(row['cat_id'])
        store_id = np.array(row['store_id'])
        state_id = np.array(row['state_id'])

        if self.transform == True:
            p_id, item_id, dept_id, cat_id, store_id, state_id = self.transforms(p_id, item_id, dept_id, cat_id, store_id, state_id)

        sample = {
            'id': id,
            'p_id': p_id,
            'item_id': item_id,
            'dept_id': dept_id,
            'cat_id': cat_id,
            'store_id': store_id,
            'state_id': state_id
        }
        return sample

    def transforms(self, p_id, item_id, dept_id, cat_id, store_id, state_id):
        p_id = torch.from_numpy(p_id).float()
        item_id = torch.from_numpy(item_id).long()
        dept_id = torch.from_numpy(dept_id).long()
        cat_id = torch.from_numpy(cat_id).long()
        store_id = torch.from_numpy(store_id).long()
        state_id = torch.from_numpy(state_id).long()
        return  p_id, item_id, dept_id, cat_id, store_id, state_id


if __name__ == "__main__":
    dataset = WalmartDataset()
    sample = dataset[0]
    print(sample['id'])
    print(sample['item_id'].__repr__)
    print(sample['state_id'].__repr__)
    from penet.models.penet import PENet
    model = PENet()

