import os
import os.path as osp
import shutil
import torch
from collections import OrderedDict
import glob


class Saver(object):
    def __init__(self, args):
        self.args = args
        self.directory = osp.join(args.out, 'run', args.model)
        self.runs = sorted(glob.glob(osp.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0
        self.experiment_dir = osp.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if args.resume is None:
            if not osp.exists(self.experiment_dir):
                os.makedirs(self.experiment_dir)
        else:
            self.experiment_dir = args.resume

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.experiment_dir, 'log.csv')):
            with open(osp.join(self.experiment_dir, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

    def save_experiment_config(self):
        logfile = osp.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['model'] = self.args.model
        p['activation'] = self.args.activation_function
        p['lr'] = self.args.lr
        p['weight_decay'] = self.args.weight_decay
        p['epochs'] = self.args.epochs
        p['batch_size'] = self.args.batch_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()

