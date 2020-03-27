import numpy as np
import torch
import torch.nn as nn


class PENet(nn.Module):
    def __init__(self, n_feature=30490, activation_function='sigmoid'):
        super(PENet, self).__init__()
        self.n_feature = n_feature
        if activation_function in ['sigmoid', 'tanh', 'relu']:
            self.activation_function = activation_function
        else:
            raise AssertionError('Activation')

        self.p2vec = nn.Linear(self.n_feature, 512)
        self.hidden = nn.Linear(512, 256)
        self.vec = nn.Linear(256, 512)

        if self.activation_function == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif self.activation_function == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        self.vec2item = nn.Linear(512, 3049)
        self.item2dept = nn.Linear(3049, 7)
        self.dept2cat = nn.Linear(7, 3)
        self.vec2store = nn.Linear(512, 10)
        self.store2state = nn.Linear(10, 3)

    def forward(self, x):
        h = self.activation(self.p2vec(x))
        h = self.activation(self.hidden(h))
        vec = self.activation(self.vec(h))

        item = self.vec2item(vec)
        dept = self.item2dept(item)
        cat = self.dept2cat(dept)
        store = self.vec2store(vec)
        state = self.store2state(store)
        return vec, item, dept, cat, store, state

        # if self.training:
        #     return item, dept, cat, store, state
        # else:
        #     return vec



if __name__ == "__main__":

    model = PENet()
    x = torch.randn(256, 30490)
    import datetime
    import pytz
    start_timestamp = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    item, dept, cat, store, state = model(x)
    total_time = (datetime.datetime.now(pytz.timezone('Asia/Tokyo')) - start_timestamp).total_seconds()

