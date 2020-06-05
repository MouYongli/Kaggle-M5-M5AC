import glob
import argparse
import yaml
import tqdm
import datetime
import pytz
import shutil
import os
import torch
import numpy as np
import torch.nn as nn
import os.path as osp
import wandb

# wandb.login(key='b9588fd28f7f6e7610d68daf4434db0bd8a3553e')
wandb.init(project="m5dataset")
wandb.watch_called = False
config = wandb.config  # Initialize config
config.seed = 666
from torch.autograd import Variable

import sys

sys.path.append('../')
from m5net.datasets.m5dataset import M5Dataset
from m5net.models.m5net import M5Net

here = osp.dirname(osp.abspath(__file__))


class Trainer(object):
    def __init__(self, cuda, model, optimizer, train_loader, val_loader, out, epochs, best_mse=np.inf,
                 interval_validate=None):
        self.cuda = cuda
        self.model = model
        self.optim = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)
        self.valid_log_headers = [
            'epoch',
            'pid',
            'sid',
            'valid/loss',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'valid_log.csv')):
            with open(osp.join(self.out, 'valid_log.csv'), 'w') as f:
                f.write(','.join(self.valid_log_headers) + '\n')
        self.loss_fn = nn.MSELoss()
        self.valid_loss_fn = nn.MSELoss(reduction="none")
        self.epoch = 0
        self.iteration = 0
        self.epochs = epochs
        self.best_mse = best_mse

    def validate(self):
        training = self.model.training
        self.model.eval()
        val_loss = 0
        for batch_idx, (pid, sid, products, events, calendar_prices, unit_sales) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader), desc='Valid iteration=%d' % self.iteration,
                ncols=80, leave=False):
            if self.cuda:
                products, events, calendar_prices, unit_sales = products.cuda(), events.cuda(), calendar_prices.cuda(), unit_sales.cuda()
            products, events, calendar_prices, unit_sales = Variable(products), Variable(events), Variable(
                calendar_prices), Variable(unit_sales)
            with torch.no_grad():
                predict_sales = self.model(products, events, calendar_prices)
            loss = self.loss_fn(predict_sales, unit_sales)
            loss_data = loss.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
            val_loss += loss_data
            # with open(osp.join(self.out, 'valid_log.csv'), 'a') as f:
            #     elapsed_time = (
            #                 datetime.datetime.now(pytz.timezone('Asia/Tokyo')) - self.timestamp_start).total_seconds()
            #     for (p, s, l) in zip(pid, sid, loss_data):
            #         log = [self.epoch, p, s, l, elapsed_time]
            #         log = map(str, log)
            #         f.write(','.join(log) + '\n')
        val_loss = val_loss / len(self.val_loader)
        wandb.log({'val_loss':val_loss})
        wandb.run.summary['vali_loss'] = val_loss
        with open(osp.join(self.out, 'valid_log.csv'), 'a') as f:
            elapsed_time = (datetime.datetime.now(pytz.timezone('Asia/Tokyo')) - self.timestamp_start).total_seconds()
            log = [self.epoch, '-', '-', val_loss, elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')
        is_best = val_loss < self.best_mse
        if is_best:
            self.best_mse = val_loss
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mse': self.best_mse,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))
            wandb.save('model_best.h5')
        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()
        for batch_idx, (pid, sid, products, events, calendar_prices, unit_sales) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader), desc='Train epoch=%d' % self.epoch,
                ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue
            self.iteration = iteration
            if self.iteration % self.interval_validate == 0:
                self.validate()
            assert self.model.training
            if self.cuda:
                products, events, calendar_prices, unit_sales = products.cuda(), events.cuda(), calendar_prices.cuda(), unit_sales.cuda()
            products, events, calendar_prices, unit_sales = Variable(products), Variable(events), Variable(
                calendar_prices), Variable(unit_sales)
            self.optim.zero_grad()
            predict_sales = self.model(products, events, calendar_prices)
            loss = self.loss_fn(predict_sales, unit_sales)
            loss.backward()
            self.optim.step()
        wandb.log({'batch':batch_idx, 'batch_loss': loss})

    def train(self):
        max_epoch = self.epochs
        for epoch in tqdm.trange(self.epoch, max_epoch, desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')

    parser.add_argument('--resume', help='checkpoint path')
    parser.add_argument('--epochs', type=int, default=30, help='epochs')

    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')

    parser.add_argument('--version', type=str, default="v1", help='version id')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--seq-length', type=int, default=1913, help='sequence length')
    parser.add_argument('--embedding-dim', type=int, default=256, help='embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=512, help='hidden dimension')
    parser.add_argument('--n-layer', type=int, default=8, help='number of layers')

    args = parser.parse_args()

    args.model = 'M5Net'
    directory = os.path.join(here, 'log', args.model, args.version, 'run')
    runs = sorted(glob.glob(os.path.join(directory, 'experiment_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    args.out = os.path.join(directory, 'experiment_{}'.format(str(run_id)))
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)
    wandb.config.update(args)
    args = wandb.config
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()
    seed = config.seed
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # 1. dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(M5Dataset(split='train', transform=True), batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(M5Dataset(split='valid', transform=True), batch_size=args.batch_size,
                                             shuffle=False, **kwargs)

    # 2. model
    model = M5Net(embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, n_layer=args.n_layer)
    start_epoch = 0
    start_iteration = 1
    best_mse = np.inf
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
        best_mse = checkpoint['best_mse']

    if cuda:
        model = model.cuda()

    # 3. optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])
    wandb.watch(model)
    trainer = Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        epochs=args.epochs,
        best_mse=best_mse,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
