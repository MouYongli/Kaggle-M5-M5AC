import torch
import torch.nn as nn
from torch.autograd import Variable
from penet.models.penet import PENet
from penet.datasets.walmartdataset import WalmartDataset

model = PENet()
train_loader= torch.utils.data.DataLoader(WalmartDataset(split='train', transform=True), batch_size=1, shuffle=True)
criterion = nn.CrossEntropyLoss()
batch_idx, sample = next(enumerate(train_loader))
p_id, item_id, dept_id, cat_id, store_id, state_id = sample['p_id'], sample['item_id'], sample['dept_id'], sample[
        'cat_id'], sample['store_id'], sample['state_id']

cuda = torch.cuda.is_available()
if cuda:
    print('Cuda is available!')
    model = model.cuda()
    p_id, item_id, dept_id, cat_id, store_id, state_id = p_id.cuda(), item_id.cuda(), dept_id.cuda(), cat_id.cuda(), store_id.cuda(), state_id.cuda()
p_id, item_id, dept_id, cat_id, store_id, state_id = Variable(p_id), Variable(item_id), Variable(dept_id), Variable(cat_id), Variable(store_id), Variable(state_id)

model.eval()
vec, item_id_pred, dept_id_pred, cat_id_pred, store_id_pred, state_id_pred = model(p_id)
loss = criterion(state_id_pred, state_id)
print(state_id)
print(state_id_pred)
print(loss)