import argparse
import tqdm
import os
import os.path as osp
import torch
from torch.autograd import Variable

import sys
sys.path.append('../../')

from penet.datasets.walmartdataset import WalmartDataset
from penet.models.penet import PENet
from penet.utils.saver import Saver
from penet.utils.metrics import Evaluator

here = osp.dirname(osp.abspath(__file__))

class Trainer(object):
    def __init__(self, args):
        self.args = args
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader = torch.utils.data.DataLoader(WalmartDataset(split='train', transform=True), batch_size=args.batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(WalmartDataset(split='valid', transform=True), batch_size=args.batch_size, shuffle=False, **kwargs)

        # Define network
        self.model = PENet()

        # Define Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Define Criterion
        # TODO

        # Define Evaluator
        # self.evaluator = Evaluator(self.nclass)

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

        # Using cuda
        self.cuda = args.cuda
        if args.cuda:
            self.model = self.model.cuda()

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        for batch_idx, sample in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='Train epoch=%d' % epoch, ncols=80, leave=False):
            iteration = batch_idx + epoch * len(self.train_loader)
            p_id, item_id, dept_id, cat_id, store_id, state_id = sample['p_id'], sample['item_id'], sample['dept_id'], sample['cat_id'], sample['store_id'], sample['state_id']
            assert self.model.training
            if self.cuda:
                p_id, item_id, dept_id, cat_id, store_id, state_id = p_id.cuda(), item_id.cuda(), dept_id.cuda(), cat_id.cuda(), store_id.cuda(), state_id.cuda()
            p_id, item_id, dept_id, cat_id, store_id, state_id = Variable(p_id), Variable(item_id), Variable(dept_id), Variable(cat_id), Variable(store_id), Variable(state_id)
            self.optimizer.zero_grad()
            item_id_pred, dept_id_pred, cat_id_pred, store_id_pred, state_id_pred = self.model(p_id)

    def validation(self, epoch):
        pass


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # training hyper params
    parser.add_argument('--model', type=str, default=None,
                        help='set the model name')
    parser.add_argument('--out', type=str, default=None,
                        metavar='N', help='out dir')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--start-epoch', type=int, default=0, help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=512,
                        metavar='N', help='input batch size for training (default: 512)')
    parser.add_argument('--activation-function', type=str, default='sigmoid',
                        choices=['sigmoid', 'tanh', 'relu'],
                        help='activation function (default: sigmoid)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    # cuda, seed
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()

    if args.model is None:
        args.model = 'penet'
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.out is None:
        args.out = here
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        args.epochs = 500
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)


if __name__ == "__main__":
    main()
